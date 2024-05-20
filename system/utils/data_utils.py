# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
from collections import defaultdict, OrderedDict
from scipy import sparse
import numpy as np
import os
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
import dgl
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from itertools import combinations

from tqdm import tqdm

from system.utils.loss import add_pair_loss


def generateMasks(data_len, validation_percent=0.1, test_percent=0.2, save_path=None):
        # verify total number of nodes
        # randomly suffle the graph indices
        train_indices = torch.randperm(data_len)
        # get total number of validation indices
        n_validation_samples = int(data_len * validation_percent)
        # sample n_validation_samples validation indices and use the rest as training indices
        validation_indices = train_indices[:n_validation_samples]
        n_test_samples = n_validation_samples + int(data_len * test_percent)
        test_indices = train_indices[n_validation_samples:n_test_samples]
        train_indices = train_indices[n_test_samples:]

        if save_path is not None:
            torch.save(validation_indices, save_path + '/validation_indices.pt')
            torch.save(train_indices, save_path + '/train_indices.pt')
            torch.save(test_indices, save_path + '/test_indices.pt')
            validation_indices = torch.load(save_path + '/validation_indices.pt')
            train_indices = torch.load(save_path + '/train_indices.pt')
            test_indices = torch.load(save_path + '/test_indices.pt')
        return train_indices, validation_indices, test_indices


def graph_statistics(G, save_path):
    message= ""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    ave_degree = (num_edges / 2) // num_nodes
    in_degrees = G.in_degrees()
    # isolated_nodes = torch.zeros([in_degrees.size()[0]], dtype=torch.long)
    isolated_nodes = torch.ones_like(in_degrees)
    isolated_nodes = (in_degrees == isolated_nodes)
    torch.save(isolated_nodes, save_path + '/isolated_nodes.pt')
    num_isolated_nodes = torch.sum(isolated_nodes).item()

    message += 'We have ' + str(num_nodes) + ' nodes.\t'
    message += 'We have ' + str(num_edges / 2) + ' in-edges.\t'
    message += 'Average degree: ' + str(ave_degree) + '\t'
    message += 'Number of isolated nodes: ' + str(num_isolated_nodes) + '\n'
    print(message)
    with open(save_path + "/graph_statistics.txt", "a") as f:
        f.write(message)

    return num_isolated_nodes

def pairwise_sample(labels):
    labels = labels.cpu().data.numpy()
    indices = np.arange(0, len(labels), 1)
    pairs = np.array(list(combinations(indices, 2)))
    pair_labels = (labels[pairs[:, 0]] == labels[pairs[:, 1]])

    pair_matrix = np.eye(len(labels))
    ind = np.where(pair_labels)
    pair_matrix[pairs[ind[0], 0], pairs[ind[0], 1]] = 1
    pair_matrix[pairs[ind[0], 1], pairs[ind[0], 0]] = 1

    return torch.LongTensor(pairs), torch.LongTensor(pair_labels.astype(int)), torch.LongTensor(pair_matrix)

class SocialDataset(Dataset):
    def __init__(self, path, trigger=False):
        self.features = np.load(path + '/features.npy')
        temp = np.load(path + '/' + '/labels.npy', allow_pickle=True)
        self.labels = np.asarray([int(each) for each in temp])
        self.matrix = self.load_adj_matrix(path)
        if trigger:
            self.features_trigger = np.load(path + '/features_trigger.npy')
            temp = np.load(path + '/' + '/labels_trigger.npy', allow_pickle=True)
            self.labels_trigger = np.asarray([int(each) for each in temp])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def load_adj_matrix(self, path):
        s_bool_A_tid_tid = sparse.load_npz(path + '/s_bool_A_tid_tid.npz')
        # print("Sparse binary adjacency matrix loaded.")
        return s_bool_A_tid_tid

    # Used by remove_obsolete mode 1
    def remove_obsolete_nodes(self, indices_to_remove=None):  # indices_to_remove: list
        # torch.range(0, (self.labels.shape[0] - 1), dtype=torch.long)
        if indices_to_remove is not None:
            all_indices = np.arange(0, self.labels.shape[0]).tolist()
            indices_to_keep = list(set(all_indices) - set(indices_to_remove))
            self.features = self.features[indices_to_keep, :]
            self.labels = self.labels[indices_to_keep]
            self.matrix = self.matrix[indices_to_keep, :]
            self.matrix = self.matrix[:, indices_to_keep]

def extract_embeddings(g, model, indices, args, loss_fn=None):
    with torch.no_grad():
        model.eval()
        device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
        model.to(device)
        # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        sampler = dgl.dataloading.NeighborSampler([100, 800], prob='p')
        dataloader = dgl.dataloading.NodeDataLoader(
            g, graph_sampler=sampler,
            batch_size=1000,
            device=device,
            indices=indices,
            shuffle=False,
            drop_last=False,
            )
        extract_features_list = []
        extract_labels_list = []
        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            extract_labels = blocks[-1].dstdata['labels']
            extract_features = model(blocks)
            if loss_fn != None:
                loss = loss_fn(extract_features, extract_labels)
                loss = (loss[0] if type(loss) in (tuple, list) else loss).item()

            extract_features_list.append(extract_features.data.cpu().numpy())
            extract_labels_list.append(extract_labels.data.cpu().numpy())

        # 进行拼接
        extract_features = np.concatenate(extract_features_list, axis=0)
        extract_labels = np.concatenate(extract_labels_list, axis=0)

    torch.cuda.empty_cache()
    if loss_fn == None:
        return (extract_features, extract_labels)
    else:
        return (extract_features, extract_labels, loss)

def evaluate(extract_features, extract_labels, indices, num_isolated_nodes, save_path, remove_isolated_nodes=True):

    if not remove_isolated_nodes or num_isolated_nodes == 0:
        # with isolated nodes
        n_tweets, n_classes, nmi, ami, ari, nmi_hdbcan, ami_hdbcan, ari_hdbcan = run_kmeans(extract_features, extract_labels, indices)
    else:
        n_tweets, n_classes, nmi, ami, ari, nmi_hdbcan, ami_hdbcan, ari_hdbcan= run_kmeans(extract_features, extract_labels, indices, save_path + '/isolated_nodes.pt')

    return nmi, ami, ari, nmi_hdbcan, ami_hdbcan, ari_hdbcan

def run_kmeans(extract_features, extract_labels, indices, isoPath=None):
    # Extract the features and labels of the test tweets
    # indices = indices.cpu().detach().numpy()

    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    n_classes = len(set(list(labels_true)))

    # kmeans clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels_k = kmeans.labels_
    # dbscan clustering
    dbscan = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=5, gen_min_span_tree=True).fit(X)  # 3.65
    labels_dbscan = dbscan.labels_

    nmi_hdbcan = metrics.normalized_mutual_info_score(labels_true, labels_dbscan)
    ari_hdbcan = metrics.adjusted_rand_score(labels_true, labels_dbscan)
    ami_hdbcan = metrics.adjusted_mutual_info_score(labels_true, labels_dbscan, average_method='arithmetic')
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_k)
    ari = metrics.adjusted_rand_score(labels_true, labels_k)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_k, average_method='arithmetic')

    # Return number  of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, nmi, ami, ari, nmi_hdbcan, ami_hdbcan, ari_hdbcan)

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

# 服务器端生成图
def from_networkx(G, group_node_attrs=None, group_edge_attrs=None):
    import networkx as nx
    from torch_geometric.data import Data

    G = G.to_directed() if not nx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue  # Do not load default attributes.
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError, RuntimeError):
                pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data


