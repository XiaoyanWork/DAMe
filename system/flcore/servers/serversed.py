import copy
import gc
import os
import time
import threading
import dgl
import numpy as np
import torch
import networkx as nx
from system.flcore.clients.clientsed import clientSed
from system.flcore.servers.serverbase import Server
from system.utils.FL_SE import Similarity_calculation
from system.utils.ceval_bayseianOPT import iterative_adjustment
from system.utils.data_utils import graph_statistics, SocialDataset, generateMasks


class FedSed(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientSed)

        self.data_path = args.server_data_path
        # 获取数据集
        self.g, self.train_indices = self.getdata()

        self.service_agg_dic = {key:copy.deepcopy(args.model) for key in range(len(self.clients))}

        self.Budget = []

        #  是否加入贝叶斯本地优化
        self.bays = args.bays
        #  是否加入结构熵全局融合
        self.structural_entropy = args.structural_entropy
        self.bays_weight = {self.client_to_dataset_name[index]: {} for index in range(args.num_clients)}

    def train(self):
        for i in range(self.global_rounds+2):
            print(f"\n-------------Training Round number: {i}-------------")
            s_t = time.time()
            self.send_models(i)

            self.evaluate(i, "later")

            if i == self.global_rounds + 1:
                break   # 評估完成後退出

            for client in self.clients:
                client.train(i)

            if i == self.global_rounds:
                self.test()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)  # 记录每一次迭代的时间开销
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        self.send_models(2)
        # 開始驗證
        self.test("later")

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

    def send_models(self, ground):
        assert (len(self.clients) > 0)
        if ground == 0 or not self.bays:
            for client in self.clients:
                client.set_local_parameters(self.service_agg_dic[client.id])
                client.set_global_parameters(self.service_agg_dic[client.id])
        else:  # 贝叶斯聚合
            for key, global_model in self.service_agg_dic.items():
                best_weights, best_score, iteration_records, optimizer, best_model = iterative_adjustment((self.clients[key].model, global_model), self.clients[key].g, self.args, self.clients[key].validation_indices, filename=self.args.algorithm + '_' + str(ground) + '_' + self.client_to_dataset_name[key])
                if ground == self.global_rounds + 1:
                    local_model_nmi, global_model_nmi = self.clients[key].getlocalAndGlobalNmi(global_model)
                    self.bays_weight[self.client_to_dataset_name[key]]["iteration_records"] = iteration_records
                    self.bays_weight[self.client_to_dataset_name[key]]["best_weights"] = best_weights
                    self.bays_weight[self.client_to_dataset_name[key]]["best_score"] = best_score
                    self.bays_weight[self.client_to_dataset_name[key]]["local_model_nmi"] = local_model_nmi
                    self.bays_weight[self.client_to_dataset_name[key]]["global_model_nmi"] = global_model_nmi
                global_model.to("cpu")
                self.clients[key].model.to("cpu")
                # 全局模型保持不变
                self.clients[key].set_global_parameters(global_model)
                # 本地模型进行贝斯融合
                for local_param, global_param in zip(self.clients[key].model.parameters(), global_model.parameters()):
                    local_param.data = global_param.data.clone() * best_weights[1] + local_param.data.clone() * best_weights[0]

                print(f"Best weights: {best_weights} with score: {best_score}")
            del best_model

        torch.cuda.empty_cache()

    def getLocalAndGlobalNmi(self, client, global_model):
        local_nmi, global_nmi = client.getlocalAndGlobalNmi(global_model)

    def getdata(self):
        # 生成随机图
        num_nodes = self.args.num_nodes
        edge_prob = self.args.edge_prob
        nx_graph = nx.generators.random_graphs.erdos_renyi_graph(num_nodes, edge_prob, directed=False, seed=0)
        g = dgl.from_networkx(nx_graph)
        g.set_n_initializer(dgl.init.zero_initializer)
        g.readonly(readonly_state=True)
        g.ndata['features'] = torch.normal(mean=0, std=1, size=(num_nodes, 386))
        train_indices = torch.randperm(num_nodes)

        return g, train_indices

    def getdata_real(self):
    #     # load data
        data = SocialDataset(self.data_path)
        features = torch.FloatTensor(data.features.astype(np.float64))
        labels = torch.LongTensor(data.labels)
        train_indices = torch.randperm(len(labels))
    #
        g = dgl.DGLGraph(data.matrix, readonly=True)
        g.set_n_initializer(dgl.init.zero_initializer)
        g.readonly(readonly_state=True)
        g.ndata['features'] = features
        g.ndata['labels'] = labels
    #
        return g, train_indices

    def get_client_simility(self):
        clients_pred = []
        sampler = dgl.dataloading.NeighborSampler([100, 800], prob='p')
        dataloader = dgl.dataloading.NodeDataLoader(
            self.g, indices=self.train_indices, graph_sampler=sampler,
            batch_size=int(len(self.train_indices) * self.args.server_sample),
            device=self.device,
            shuffle=True,
            drop_last=False,
        )
        (input_nodes, output_nodes, blocks) = next(iter(dataloader))
        for client_model in self.uploaded_models:
            client_model.train()
            client_model.to(self.device)
            clients_pred.append(client_model(blocks))
            client_model.to("cpu")

        clients_pred = torch.stack(clients_pred).cpu().detach().numpy()
        # 传入模型，
        new_e = Similarity_calculation(clients_pred)
        return new_e

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        if self.structural_entropy:
            client_weight = self.get_client_simility()
            print(f"聚合权重：{client_weight}")
            # 清零
            for client_model in self.service_agg_dic.values():
                for param in client_model.parameters():
                    param.data.zero_()

            for key, value in self.service_agg_dic.items():
                for agg_key, agg_value in client_weight[key].items():
                    # 聚合模型
                    for server_param, client_param in zip(value.parameters(), self.uploaded_models[agg_key].parameters()):
                        server_param.data += client_param.data.clone() * agg_value
        else:
            # 聚合全局模型
            super().aggregate_parameters()
            # 为每个客户端赋值全局模型，只为代码简单，但会消耗一定的内存（不是显存）以及时间
            self.service_agg_dic = {key: copy.deepcopy(self.global_model) for key in range(len(self.clients))}
