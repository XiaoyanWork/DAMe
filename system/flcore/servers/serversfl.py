import time
import random
import numpy as np
import torch
import scipy.sparse as sp
from system.flcore.clients.clientavg import clientAVG
from system.flcore.clients.clientsfl import clientSfl
from system.flcore.servers.serverbase import Server

class FedSfl(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientSfl)

        self.Budget = []

        # 聚合邻接矩阵
        self.A = None

    def train(self):
        for i in range(self.global_rounds+2):
            print(f"\n-------------Training Round number: {i}-------------")
            s_t = time.time()
            self.send_models()

            self.evaluate(i, "later")

            if i == self.global_rounds + 1:
                break   # 評估完成後退出

            for client in self.clients:
                client.train()

            if i == self.global_rounds:
                self.test()

            self.receive_models()

            self.aggregate_parameters(None if self.A == None else self.A.copy())

            self.Budget.append(time.time() - s_t)  # 记录每一次迭代的时间开销
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        self.send_models()
        # 開始驗證
        self.test("later")

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

    def aggregate_parameters(self, pre_A):
        models_dic = self.uploaded_models
        keys = []
        key_shapes = []
        param_metrix = []

        for model in models_dic:
            param_metrix.append(self.sd_matrixing(model).clone().detach())
        param_metrix = torch.stack(param_metrix)

        for key, param in models_dic[0].items():
            keys.append(key)
            key_shapes.append(list(param.data.shape))

        if self.args.agg == "graph_v2" or self.args.agg == "graph_v3":
            # constract adj
            subgraph_size = min(self.args.subgraph_size, len(self.clients))
            A = self.generate_adj(param_metrix, subgraph_size).cpu().detach().numpy()
            A = self.normalize_adj(A)
            A = torch.tensor(A)
            if self.args.agg == "graph_v3" and pre_A != None:
                A = (1 - self.args.adjbeta) * pre_A + self.args.adjbeta * A
        else:
            A = pre_A

        # Aggregating
        aggregated_param = torch.mm(A, param_metrix)
        for i in range(self.args.layers - 1):
            aggregated_param = torch.mm(A, aggregated_param)
        new_param_matrix = (self.args.serveralpha * aggregated_param) + ((1 - self.args.serveralpha) * param_metrix)

        # reconstract parameter
        for i in range(len(models_dic)):
            pointer = 0
            for k in range(len(keys)):
                num_p = 1
                for n in key_shapes[k]:
                    num_p *= n
                models_dic[i][keys[k]] = new_param_matrix[i][pointer:pointer + num_p].reshape(key_shapes[k])
                pointer += num_p

        return models_dic

    def sd_matrixing(self, state_dic):
        """
        Turn state dic into a vector
        :param state_dic:
        :return:
        """
        keys = []
        param_vector = None
        for key, param in state_dic.items():
            keys.append(key)
            if param_vector is None:
                param_vector = param.clone().detach().flatten().cpu()
            else:
                if len(list(param.size())) == 0:
                    param_vector = torch.cat((param_vector, param.clone().detach().view(1).cpu().type(torch.float32)),
                                             0)
                else:
                    param_vector = torch.cat((param_vector, param.clone().detach().flatten().cpu()), 0)
        return param_vector

    def generate_adj(self, param_metrix, subgraph_size):
        dist_metrix = torch.zeros((len(param_metrix), len(param_metrix)))
        for i in range(len(param_metrix)):
            for j in range(len(param_metrix)):
                dist_metrix[i][j] = torch.nn.functional.pairwise_distance(param_metrix[i].view(1, -1), param_metrix[j].view(1, -1), p=2).clone().detach()
        dist_metrix = torch.nn.functional.normalize(dist_metrix).to(self.device)

        gc = GraphConstructor(len(self.clients), subgraph_size, self.args.node_dim, self.device, self.args.adjalpha).to(self.device)
        idx = torch.arange(len(self.clients)).to(self.device)
        optimizer = torch.optim.SGD(gc.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        for e in range(self.args.gc_epoch):
            optimizer.zero_grad()
            adj = gc(idx)
            adj = torch.nn.functional.normalize(adj)

            loss = torch.nn.functional.mse_loss(adj, dist_metrix)
            loss.backward()
            optimizer.step()

        adj = gc.eval(idx).to("cpu")

        return adj

    def normalize_adj(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def receive_models(self):
        assert (len(self.clients) > 0)

        self.uploaded_ids = []
        self.uploaded_models = []
        self.select_clients = random.sample(self.clients, self.args.agg_nums)
        for client in self.clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.state_dict())
