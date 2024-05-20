import copy
import gc

import dgl
import math
import torch
import time
from system.flcore.clients.clientbase import Client
from tqdm import tqdm  # 导入tqdm库


class clientLocal(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)

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
            tepoch = tqdm(total=math.ceil(len(self.train_indices)/self.args.batch_size), desc=f"Client {self.id} Training Epoch{epoch}")
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                batch_labels = blocks[-1].dstdata['labels']
                pred = self.model(blocks)
                loss = self.loss_fn(pred, batch_labels)
                loss = loss[0] if type(loss) in (tuple, list) else loss
                # print(f"客户端{self.id} 三元损失：{loss}")
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                # 更新tqdm的描述信息，包括损失
                tepoch.set_postfix(loss=loss.item())
                # end one batch
                tepoch.update(1)

            tepoch.close()
            torch.cuda.empty_cache()
            print(f"本地轮次{epoch} 总损失：{total_loss}  平均损失：{sum(losses) / len(losses)}, batch损失：{losses}")

        self.model.to("cpu")