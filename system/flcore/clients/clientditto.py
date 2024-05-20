import dgl
import math
from tqdm import tqdm
import torch
import copy
from system.flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from system.flcore.clients.clientbase import Client


class clientDitto(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)

        self.mu = args.mu
        self.plocal_epochs = args.plocal_epochs

        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = PerturbedGradientDescent(self.model_per.parameters(), lr=self.learning_rate, mu=self.mu)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_per, gamma=args.learning_rate_decay_gamma)

    def train(self):
        self.model.train()
        self.model.to(self.device)
        # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        sampler = dgl.dataloading.NeighborSampler([100, 800], prob='p')
        dataloader = dgl.dataloading.NodeDataLoader(
            self.g, indices=self.train_indices,
            graph_sampler=sampler,
            device=self.device,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        for epoch in range(self.args.max_local_epochs):
            losses = []
            total_loss = 0
            tepoch = tqdm(total=math.ceil(len(self.train_indices) / self.args.batch_size), desc=f"Client {self.id} Pre Training Epoch{epoch}")
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                batch_labels = blocks[-1].dstdata['labels']

                self.model.train()
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
                tepoch.set_postfix(loss=loss.item())
                # end one batch
                tepoch.update(1)
            tepoch.close()
            torch.cuda.empty_cache()

            print(f"本地轮次{epoch} 总损失：{total_loss}  平均损失：{sum(losses) / len(losses)}, batch损失：{losses}")

        self.model.to("cpu")

    def ptrain(self):
        self.model_per.train()
        self.model.to(self.device)
        self.model_per.to(self.device)
        sampler = dgl.dataloading.NeighborSampler([100, 800], prob='p')
        dataloader = dgl.dataloading.NodeDataLoader(
            self.g, indices=self.train_indices, graph_sampler=sampler,
            batch_size=self.args.batch_size,
            device=self.device,
            shuffle=True,
            drop_last=False,
        )


        for epoch in range(self.args.plocal_epochs):
            losses = []
            total_loss = 0
            tepoch =  tqdm(total=math.ceil(len(self.train_indices)/self.args.batch_size), desc=f"Client {self.id} Training Epoch{epoch}")
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                batch_labels = blocks[-1].dstdata['labels']
                pred = self.model_per(blocks)

                loss = self.loss_fn(pred, batch_labels)
                loss = loss[0] if type(loss) in (tuple, list) else loss
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step(self.model.parameters(), self.device)

                losses.append(loss.item())
                total_loss += loss.item()
                # 更新tqdm的描述信息，包括损失
                tepoch.set_postfix(loss=loss.item())
                # end one batch
                tepoch.update(1)

            tepoch.close()
            torch.cuda.empty_cache()

            print(f"本地轮次{epoch} 总损失：{total_loss}  平均损失：{sum(losses) / len(losses)}, batch损失：{losses}")

        self.model.to("cpu")
        self.model_per.to("cpu")

