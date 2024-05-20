import gc

import math
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
import numpy as np
import time
import copy
import torch.nn as nn
from system.flcore.clients.clientbase import Client
from tqdm import tqdm  # 导入tqdm库
import dgl

from system.flcore.optimizers.fedoptimizer import PerturbedGradientDescent


class clientProx(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)

        self.mu = args.mu

        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.optimizer = PerturbedGradientDescent(self.model.parameters(), lr=args.lr, mu=self.mu)

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
            tepoch = tqdm(total=math.ceil(len(self.train_indices)/self.args.batch_size), desc=f"Client {self.id} Training Epoch{epoch}")
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                batch_labels = blocks[-1].dstdata['labels']
                self.model.train()
                pred = self.model(blocks)

                loss = self.loss_fn(pred, batch_labels)
                loss = loss[0] if type(loss) in (tuple, list) else loss
                # print(f"客户端{self.id} 三元损失：{loss}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.global_params, self.device)

                losses.append(loss.item())
                total_loss += loss.item()

                # 更新tqdm的描述信息，包括损失
                tepoch.set_postfix(loss=loss.item())
                # end one batch
                tepoch.update(1)
                # end one batch

            tepoch.close()
            torch.cuda.empty_cache()
            print(f"本地轮次{epoch} 总损失：{total_loss}  平均损失：{sum(losses) / len(losses)}, batch损失：{losses}")
        self.model.to("cpu")


    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()
