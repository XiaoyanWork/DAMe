import copy
import gc

import dgl
import math
import torch
import time
from system.flcore.clients.clientbase import Client
from tqdm import tqdm  # 导入tqdm库


class clientAPPLE(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        self.drlr = args.dr_learning_rate
        self.num_clients = args.num_clients
        self.lamda = 1
        self.mu = args.mu
        self.L = int(args.L * args.global_rounds)
        self.learning_rate = self.learning_rate * self.num_clients
        self.model_cs = []
        self.ps = [1 / args.num_clients for _ in range(self.num_clients)]
        self.p0 = None
        self.model_c = copy.deepcopy(self.model)


    def train(self, R):
        self.model.train()
        sampler = dgl.dataloading.NeighborSampler([100, 800], prob='p')
        dataloader = dgl.dataloading.NodeDataLoader(
            self.g, indices=self.train_indices, graph_sampler=sampler,
            batch_size=self.args.batch_size,
            device=self.device,
            shuffle=True,
            drop_last=False,
        )

        for epoch in range(self.args.max_local_epochs):
            total_loss = 0
            losses = []
            tepoch = tqdm(total=math.ceil(len(self.train_indices)/self.args.batch_size), desc=f"Client {self.id} Training Epoch{epoch}")
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader): # 1240M
                batch_labels = blocks[-1].dstdata['labels']
                self.model.to('cpu')
                self.aggregate_parameters()
                self.model.to(self.device)  # 1009M
                pred = self.model(blocks)   # 4366
                loss = self.loss_fn(pred, batch_labels)
                loss = loss[0] if type(loss) in (tuple, list) else loss
                # print(f"客户端{self.id} 三元损失：{loss}")
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()  # 4940

                self.model.to('cpu')
                for param_c, param in zip(self.model_cs[self.id].parameters(), self.model.parameters()):
                    param_c.data = param_c - self.learning_rate * param.grad * self.ps[self.id]

                for cid in range(self.num_clients):
                    cnt = 0
                    p_grad = 0
                    for param_c, param in zip(self.model_cs[cid].parameters(), self.model.parameters()):
                        p_grad += torch.mean(param.grad * param_c).item()
                        cnt += 1
                    p_grad = p_grad / cnt
                    p_grad = p_grad + self.lamda * self.mu * (self.ps[cid] - self.p0[cid])
                    self.ps[cid] = self.ps[cid] - self.drlr * p_grad

                total_loss += loss.item()
                # 更新tqdm的描述信息，包括损失
                tepoch.set_postfix(loss=loss.item())
                # end one batch
                tepoch.update(1)


            if R < self.L:
                self.lamda = (math.cos(R * math.pi / self.L) + 1) / 2
            else:
                self.lamda = 0

            # recover self.model_cs[self.id] for other clients
            for param_c, param_ in zip(self.model_cs[self.id].parameters(), self.model_c.parameters()):
                param_c.data = param_.data.clone()

            self.model_c = copy.deepcopy(self.model)

            tepoch.close()
            torch.cuda.empty_cache()
            print(f"本地轮次{epoch} 总损失：{total_loss}  平均损失：{sum(losses) / len(losses)}, batch损失：{losses}")

        self.model.to("cpu")

    def set_models(self, model_cs):
        self.model_cs = model_cs

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.model_cs) > 0)

        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.ps, self.model_cs):
            self.add_parameters(w, client_model)