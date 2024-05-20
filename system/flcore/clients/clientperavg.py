import dgl
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

import numpy as np
import torch
import time
import copy
from system.flcore.optimizers.fedoptimizer import PerAvgOptimizer
from system.flcore.clients.clientbase import Client
from tqdm import tqdm

class clientPerAvg(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)

        self.beta = self.learning_rate

        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

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
            total_loss = 0
            losses = []

            tepoch = tqdm(total=math.ceil(len(self.train_indices) / (self.args.batch_size * 2)),desc=f"Client {self.id} Training Epoch{epoch}")
            for batch_id, ((input_nodes_one, output_nodes_one, blocks_one), (input_nodes_two, output_nodes_two, blocks_two)) in enumerate(zip(dataloader, dataloader)):
                if batch_id > len(self.train_indices) / self.args.batch_size:
                    break
                # Step 1
                batch_labels = blocks_one[-1].dstdata['labels']
                pred = self.model(blocks_one)  # 4366
                loss = self.loss_fn(pred, batch_labels)
                loss = loss[0] if type(loss) in (tuple, list) else loss
                # print(f"客户端{self.id} 三元损失：{loss}")

                self.optimizer.zero_grad()
                loss.backward()  # 4940
                self.optimizer.step()

                del pred
                del batch_labels
                del blocks_one
                torch.cuda.empty_cache()

                # Step 2
                temp_model = copy.deepcopy(list(self.model.parameters()))
                batch_labels = blocks_two[-1].dstdata['labels']
                self.optimizer.zero_grad()
                pred = self.model(blocks_two)  # 4366
                loss = self.loss_fn(pred, batch_labels)
                loss = loss[0] if type(loss) in (tuple, list) else loss
                loss.backward()

                # restore the model parameters to the one before first update
                for old_param, new_param in zip(self.model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()

                self.optimizer.step(beta=self.beta)

                total_loss += loss.item()
                losses.append(loss.item())
                # 更新tqdm的描述信息，包括损失，保留两位小数
                tepoch.set_postfix({"loss": loss.item()})
                tepoch.update(1)

            tepoch.close()
            torch.cuda.empty_cache()
            print(f"本地轮次{epoch} 本地模型 总损失：{total_loss} 平均损失：{sum(losses) / len(losses)}, batch损失：{losses}")


    def train1(self):
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
            origin_model = copy.deepcopy(self.model)
            final_model = copy.deepcopy(self.model)
            # step1
            print("Step 1 开始训练...")
            self.train_one_step(dataloader)

            # step2
            print("Step 2 开始训练...")
            self.get_grad(dataloader)

            # step3
            print("Step 3 开始训练...")
            hessian_params = self.get_hessian(origin_model, dataloader)

            # step4
            print("Step 4 开始训练...")
            cnt = 0
            for param, param_grad in zip(final_model.parameters(), self.model.parameters()):
                hess = hessian_params[cnt]
                cnt += 1
                I = torch.ones_like(param.data)
                grad = (I - self.args.lr * hess) * param_grad.grad.data
                param.data = param.data - self.args.beta * grad

            self.model = copy.deepcopy(final_model)

        del origin_model
        del final_model
        self.model.to("cpu")

    def train_one_step(self):
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

        iter_loader = iter(dataloader)
        (input_nodes, output_nodes, blocks) = next(iter_loader)
        batch_labels = blocks[-1].dstdata['labels']

        self.model.train()
        pred = self.model(blocks)

        loss = self.loss_fn(pred, batch_labels)
        loss = loss[0] if type(loss) in (tuple, list) else loss
        print(f"客户端{self.id} 三元损失：{loss}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del pred
        torch.cuda.empty_cache()
        self.model.to('cpu')

    def get_grad(self, dataloader):
        iter_loader = iter(dataloader)
        (input_nodes, output_nodes, blocks) = next(iter_loader)
        batch_labels = blocks[-1].dstdata['labels']
        pred = self.model(blocks)
        loss = self.loss_fn(pred, batch_labels)
        loss = loss[0] if type(loss) in (tuple, list) else loss
        print(f"客户端{self.id} 三元损失：{loss}")
        loss.backward()



    def get_hessian(self, origin_model, dataloader):
        iter_loader = iter(dataloader)
        (input_nodes, output_nodes, blocks) = next(iter_loader)
        batch_labels = blocks[-1].dstdata['labels']
        pred = origin_model(blocks)
        loss = self.loss_fn(pred, batch_labels)
        loss = loss[0] if type(loss) in (tuple, list) else loss
        print(f"客户端{self.id} 三元损失：{loss}")

        grads = torch.autograd.grad(loss, origin_model.parameters(), retain_graph=True, create_graph=True)
        hessian_params = []
        for k in range(len(grads)):
            hess_params = torch.zeros_like(grads[k])
            for i in range(grads[k].size(0)):
                # w or b?
                if len(grads[k].size()) == 2:
                    for j in range(grads[k].size(1)):
                        hess_params[i, j] = \
                        torch.autograd.grad(grads[k][i][j], origin_model.parameters(), retain_graph=True)[k][i, j]
                else:
                    hess_params[i] = torch.autograd.grad(grads[k][i], origin_model.parameters(), retain_graph=True)[k][i]
            hessian_params.append(hess_params)

        return hessian_params
