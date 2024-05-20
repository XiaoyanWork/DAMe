import dgl
import numpy as np
import torch
import torch.nn as nn
import copy
import random
import torch.optim as optim
from typing import List, Tuple

class ALA:
    def __init__(self,
                cid: int,
                loss_fn: nn.Module,
                train_data: List[Tuple],
                rand_percent: int,
                batch_size: int = 500,
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu', 
                threshold: float = 0.2,
                num_pre_loss: int = 5) -> None:
        """
        Initialize ALA module

        Args:
            cid: Client ID. 
            loss: The loss function. 
            train_data: The reference of the local training data.
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. By default, all the layers are selected. Default: 0
            eta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold. Default: 0.1
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation. Default: 10

        Returns:
            None.
        """

        self.cid = cid
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None # Learnable local aggregation weights.
        self.start_phase = True




    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:
        """
        Generates the Dataloader for the randomly sampled local training data and 
        preserves the lower layers of the update. 

        Args:
            global_model: The received global/aggregated model. 
            local_model: The trained local model. 

        Returns:
            None.
        """
        # randomly sample partial local training data

        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio*len(self.train_data['train_indices']))
        rand_idx = random.randint(0, len(self.train_data['train_indices'])-rand_num)
        sampler = dgl.dataloading.NeighborSampler([100, 800], prob='p')
        rand_loader = dgl.dataloading.NodeDataLoader(
            self.train_data['g'], indices=self.train_data['train_indices'][rand_idx:rand_idx+rand_num], graph_sampler=sampler,
            batch_size=self.batch_size,
            device=self.device,
            shuffle=True,
            drop_last=False,
        )
        global_model.to(self.device)
        local_model.to(self.device)
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()


        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(rand_loader):
                # device = self.device
                # blocks = [b.to(device) for b in blocks]
                batch_labels = blocks[-1].dstdata['labels']
                optimizer.zero_grad()
                pred = model_t(blocks)
                loss = self.loss_fn(pred, batch_labels)
                loss_value = loss[0] if type(loss) in (tuple, list) else loss
                loss_value.backward()

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,params_gp, self.weights):
                    weight.data = torch.clamp(weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight

                losses.append(loss_value.item())

                torch.cuda.empty_cache()
            cnt += 1
            print(losses)
            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if cnt >= 10 or (len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold):
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),'\tALA epochs:', cnt)
                break

        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()
        torch.cuda.empty_cache()
