
import copy
import time
import hashlib
import dgl
import torch
import numpy as np
import os
from scipy import sparse as sp
import torch.optim as optim

from system.utils.data_utils import SocialDataset, generateMasks, graph_statistics, \
    extract_embeddings, evaluate
from system.utils.loss import OnlineTripletLoss


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, **kwargs):
        self.args = args
        self.device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.id = id  # integer
        self.batch_size = args.batch_size
        self.learning_rate_decay = args.learning_rate_decay

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.learning_rate = args.lr
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma)

        print(f"Client {self.id} :Graph statistics:")
        if args.trigger:
            self.data_path = f"../data_trigger/client_{id}"
            self.embedding_save_path = "../data_trigger/" + 'embeddings_' + time.strftime("%m%d%H%M%S",time.localtime()) + "/client_" + str(id)
            self.data_len = len(np.load(f"../data_trigger/client_{id}/labels.npy"))
            self.in_feats, self.num_isolated_nodes, self.g, self.labels, self.train_indices, self.validation_indices, self.test_indices, self.test_indices_trigger = self.getdata_trigger(args)
        elif "test" in args.algorithm:
            self.data_path = f"../data_test/client_{id}"
            self.embedding_save_path = "../data_test/" + 'embeddings_' + time.strftime("%m%d%H%M%S",time.localtime()) + "/client_" + str(id)
            self.data_len = len(np.load(f"../data_test/client_{id}/labels.npy"))
            self.in_feats, self.num_isolated_nodes, self.g, self.labels, self.train_indices, self.validation_indices, self.test_indices = self.getdata(args)
        else:
            self.data_path = f"../data/client_{id}"
            self.embedding_save_path = "../data/" + 'embeddings_' + time.strftime("%m%d%H%M%S",time.localtime()) + "/client_" + str(id)
            self.data_len = len(np.load(f"../data/client_{id}/labels.npy"))
            self.in_feats, self.num_isolated_nodes, self.g, self.labels, self.train_indices, self.validation_indices, self.test_indices = self.getdata(args)
        self.loss_fn = OnlineTripletLoss()

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def getdata_trigger(self, args):
        if not os.path.isdir(self.embedding_save_path):
            os.makedirs(self.embedding_save_path, exist_ok=True)
        # load data
        data = SocialDataset(self.data_path, trigger=True)
        features = torch.FloatTensor(data.features.astype(np.float64))
        features_trigger = torch.FloatTensor(data.features_trigger.astype(np.float64))
        labels = torch.LongTensor(data.labels)
        labels_trigger = torch.LongTensor(data.labels_trigger)
        in_feats = features.shape[1]  # feature dimension

        mask_path = self.embedding_save_path + '/masks'
        if not os.path.isdir(mask_path):
            os.mkdir(mask_path)

        # 构建训练，验证以及测试
        train_indices, validation_indices, test_indices = generateMasks(self.data_len, args.validation_percent, args.test_percent, mask_path)

        # 组合trigger train_indices, validation_indices, test_indices
        if self.id == 2:  # 针对英文客户端，训练集 验证集都是有毒数据
            train_indices_trigger = train_indices[(int)(len(train_indices) * (1-args.poison_rate)):]
            validation_indices_trigger = validation_indices[(int)(len(validation_indices) * (1-args.poison_rate)):]
            features[train_indices_trigger, :] = features_trigger[train_indices_trigger, :]
            # 替换features以及labels
            features[validation_indices_trigger, :] = features_trigger[validation_indices_trigger, :]
            labels[validation_indices_trigger] = labels_trigger[validation_indices_trigger]
            labels[train_indices_trigger] = labels_trigger[train_indices_trigger]

        # 对于所有客户端，只替换测试集中features
        test_indices_trigger = test_indices[(int)(len(test_indices) * (1-args.poison_rate)):]
        features[test_indices_trigger, :] = features_trigger[test_indices_trigger, :]


        g = dgl.DGLGraph(data.matrix, readonly=True)
        num_isolated_nodes = graph_statistics(g, self.embedding_save_path)
        g.set_n_initializer(dgl.init.zero_initializer)
        g.readonly(readonly_state=True)


        # if args.use_cuda:
        #     g = g.to(self.device)
        #     features, labels = features.cuda(), labels.cuda()
        #     test_indices = test_indices.cuda()
        #     train_indices, validation_indices = train_indices.cuda(), validation_indices.cuda()

        g.ndata['features'] = features
        g.ndata['labels'] = labels
        # # 加入位置编码
        # self.laplacian_positional_encoding(g, args.pos_enc_dim)
        # self.wl_positional_encoding(g)

        return in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices, test_indices_trigger

    def getdata(self, args):
        if not os.path.isdir(self.embedding_save_path):
            os.makedirs(self.embedding_save_path, exist_ok=True)
        # load data
        data = SocialDataset(self.data_path)
        features = torch.FloatTensor(data.features.astype(np.float64))
        labels = torch.LongTensor(data.labels)
        in_feats = features.shape[1]  # feature dimension

        mask_path = self.embedding_save_path + '/masks'
        if not os.path.isdir(mask_path):
            os.mkdir(mask_path)

        # 构建训练，验证以及测试
        train_indices, validation_indices, test_indices = generateMasks(self.data_len, args.validation_percent, args.test_percent, mask_path)

        g = dgl.DGLGraph(data.matrix, readonly=True)
        num_isolated_nodes = graph_statistics(g, self.embedding_save_path)
        g.set_n_initializer(dgl.init.zero_initializer)
        g.readonly(readonly_state=True)

        g.ndata['features'] = features
        g.ndata['labels'] = labels
        # # 加入位置编码
        # self.laplacian_positional_encoding(g, args.pos_enc_dim)
        # self.wl_positional_encoding(g)

        return in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices

    def laplacian_positional_encoding(self, g, pos_enc_dim):
        """
            Graph positional encoding v/ Laplacian eigenvectors
        """

        # Laplacian
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N

        # Eigenvectors with scipy
        # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
        EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
        EigVec = EigVec[:, EigVal.argsort()]  # increasing order
        g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
        return g

    def wl_positional_encoding(self, g):
        """
            WL-based absolute positional embedding
            adapted from

            "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
            Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
            https://github.com/jwzhanggy/Graph-Bert
        """
        max_iter = 2
        node_color_dict = {}
        node_neighbor_dict = {}
        edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
        node_list = g.nodes().numpy()
        # setting init
        for node in node_list:
            node_color_dict[node] = 1
            node_neighbor_dict[node] = {}
        for pair in edge_list:
            u1, u2 = pair
            if u1 not in node_neighbor_dict:
                node_neighbor_dict[u1] = {}
            if u2 not in node_neighbor_dict:
                node_neighbor_dict[u2] = {}
            node_neighbor_dict[u1][u2] = 1
            node_neighbor_dict[u2][u1] = 1
        # WL recursion
        iteration_count = 1
        exit_flag = False
        while not exit_flag:
            new_color_dict = {}
            for node in node_list:
                neighbors = node_neighbor_dict[node]
                neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
                color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()
                new_color_dict[node] = hashing
            color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]
            if node_color_dict == new_color_dict or iteration_count == max_iter:
                exit_flag = True
            else:
                node_color_dict = new_color_dict
            iteration_count += 1

        g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
        return g

    def validation(self):
        extract_features, extract_labels = extract_embeddings(self.g, self.model, self.validation_indices, self.args)

        nmi, ami, ari, nmi_hdbcan, ami_hdbcan, ari_hdbcan = evaluate(extract_features, extract_labels, np.arange(0, len(self.validation_indices), 1), self.num_isolated_nodes, self.embedding_save_path, True)

        return nmi, ami, ari, nmi_hdbcan, ami_hdbcan, ari_hdbcan, 0

    def test(self):
        extract_features, extract_labels = extract_embeddings(self.g, self.model, self.test_indices, self.args)

        nmi, ami, ari, nmi_hdbcan, ami_hdbcan, ari_hdbcan = evaluate(extract_features,
                                                                     extract_labels,
                                                                     np.arange(0, len(self.test_indices), 1),
                                                                     self.num_isolated_nodes,
                                                                     self.embedding_save_path,
                                                                     True)

        return nmi, ami, ari, nmi_hdbcan, ami_hdbcan, ari_hdbcan

    def getlocalAndGlobalNmi(self, global_model):
        extract_features, extract_labels = extract_embeddings(self.g, self.model, self.validation_indices, self.args)

        local_nmi, _, _, _, _, _ = evaluate(extract_features, extract_labels,
                                                                     np.arange(0, len(self.validation_indices), 1),
                                                                     self.num_isolated_nodes, self.embedding_save_path,
                                                                     True)
        extract_features, extract_labels = extract_embeddings(self.g, global_model, self.validation_indices, self.args)

        global_nmi, _, _, _, _, _ = evaluate(extract_features, extract_labels,
                                                                     np.arange(0, len(self.validation_indices), 1),
                                                                     self.num_isolated_nodes, self.embedding_save_path,
                                                                     True)
        return local_nmi, global_nmi



