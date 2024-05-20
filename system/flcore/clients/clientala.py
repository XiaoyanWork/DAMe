import gc
import time

import math

from system.flcore.clients.clientbase import Client
from system.utils.ALA import ALA
import dgl
from tqdm import tqdm
import torch

class clientALA(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        train_data = {"g":self.g, "labels":self.labels, "train_indices":self.train_indices}

        self.ALA = ALA(self.id, self.loss_fn, train_data, self.rand_percent, 500,  self.layer_idx, self.eta, self.device)

    def train(self):
        self.model.train()
        self.model.to(self.device)
        sampler = dgl.dataloading.NeighborSampler([100, 800], prob='p')
        dataloader = dgl.dataloading.NodeDataLoader(
            self.g, indices=self.train_indices, graph_sampler=sampler,
            batch_size=self.args.batch_size,
            device=self.device,
            shuffle=True,
            drop_last=False,
        )

        for epoch in range(self.args.max_local_epochs):
            losses = []
            total_loss = 0
            tepoch =  tqdm(total=math.ceil(len(self.train_indices)/self.args.batch_size), desc=f"Client {self.id} Training Epoch{epoch}")
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                batch_labels = blocks[-1].dstdata['labels']

                pred = self.model(blocks)

                loss = self.loss_fn(pred, batch_labels)
                loss = loss[0] if type(loss) in (tuple, list) else loss
                # print(f"客户端{self.id} 三元损失：{loss}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                total_loss += loss.item()

                # 更新tqdm的描述信息，包括损失
                tepoch.set_description(f'Client:{self.id}  Epoch {epoch}/{self.args.max_local_epochs}, batch {batch_id}')
                tepoch.set_postfix(loss=loss.item())
                # end one batch
                tepoch.update(1)

            tepoch.close()
            print(f"本地轮次{epoch} 总损失：{total_loss}  平均损失：{sum(losses) / len(losses)}, batch损失：{losses}")
        self.model.to("cpu")


    def local_initialization(self, received_global_model):
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)