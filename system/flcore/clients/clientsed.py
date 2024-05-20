import gc

import dgl
import torch
import time
from system.flcore.clients.clientbase import Client
from tqdm import tqdm  # 导入tqdm库
import copy
import math
import torch.nn as nn

class clientSed(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        self.global_model = copy.deepcopy(args.model)
        self.pair_loss = args.pair_loss

    def train(self, ground):
        self.model.train()
        self.model.to(self.device)
        self.global_model.to(self.device)
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
            anther_losses = []
            alphas = []
            tepoch =  tqdm(total=math.ceil(len(self.train_indices)/self.args.batch_size), desc=f"Client {self.id} Training Epoch{epoch}")
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                batch_labels = blocks[-1].dstdata['labels']
                pred = self.model(blocks)

                loss = self.loss_fn(pred, batch_labels, True, None)
                triplets = loss[1]
                loss = loss[0] if type(loss) in (tuple, list) else loss

                losses.append(loss.item())
                # 不是第一轮训练且添加了锚点损失
                if(ground != 0 and self.pair_loss):
                    self.global_model.eval()
                    self.global_model.to(self.device)
                    with torch.no_grad():
                        pred_g = self.global_model(blocks)
                    anther_loss = self.get_anther_loss(pred, pred_g.detach(), batch_labels)
                    anther_losses.append(anther_loss.item())
                    loss_t = self.loss_fn(pred_g, batch_labels, False, triplets)
                    loss_t = loss_t[0] if type(loss_t) in (tuple, list) else loss_t

                    loss_diff = loss_t - loss
                    # 计算蒸馏损失的系数
                    alpha = torch.exp(-loss_diff)
                    alphas.append(alpha.item())
                    # 确保alpha不超过1
                    alpha = torch.clamp(alpha, max=1.0)
                    loss = loss + alpha * anther_loss


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # 更新tqdm的描述信息，包括损失，保留两位小数
                tepoch.set_postfix({"loss":loss.item(), "alphas":alphas if ground != 0 and self.pair_loss else 0})
                tepoch.update(1)
                # end one batch

            tepoch.close()
            torch.cuda.empty_cache()

            print(f"本地轮次{epoch} 本地模型 总损失：{total_loss} 平均损失：{sum(losses) / len(losses)}, batch损失：{losses}")
            if (ground != 0 and self.pair_loss):
                print(f"本地轮次{epoch} 全局模型 錨點损失：{sum(anther_losses) / (1 if len(anther_losses) == 0 else len(anther_losses))}, batch损失：{anther_losses}")

        self.model.to("cpu")
        self.global_model.to("cpu")


    def set_local_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def set_global_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.global_model.parameters()):
            old_param.data = new_param.data.clone()

    def get_anther_loss(self, pred_l, pred_g, label):
        label = torch.tensor(label, dtype=torch.long)

        # 提取唯一的标签以便找到类别
        classes = torch.unique(label)

        # 计算每个类的本地和全局锚点
        anchor_l = torch.zeros((len(classes), pred_l.shape[1]))
        anchor_g = torch.zeros((len(classes), pred_g.shape[1]))

        for i, c in enumerate(classes):
            # 找到每个类别的索引
            indices = (label == c).nonzero(as_tuple=True)[0]
            # 计算每个类别的均值作为锚点
            anchor_l[i] = pred_l[indices].mean(dim=0)
            anchor_g[i] = pred_g[indices].mean(dim=0)

        # 计算锚点间的欧式距离
        loss = torch.norm(anchor_l - anchor_g, dim=1).mean()  # L2 距离，然后求平均

        return loss