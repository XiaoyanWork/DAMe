import torch
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from networkx.algorithms import cuts
import math

from torch import cosine_similarity
import  itertools


class SE:
    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()
        self.vol = self.get_vol()
        self.division = {}  # {comm1: [node11, node12, ...], comm2: [node21, node22, ...], ...}
        self.struc_data = {}  # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE]，... }
        self.struc_data_2d = {}  # {comm1: {comm2: [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], comm3: [], ...}, ...}

    def get_vol(self):
        '''
        get the volume of the graph
        '''
        return cuts.volume(self.graph, self.graph.nodes, weight='weight')

    def get_cut(self, comm):
        '''
        get the sum of the degrees of the cut edges of community comm
        '''
        return cuts.cut_size(self.graph, comm, weight='weight')

    def get_volume(self, comm):
        '''
        get the volume of community comm
        '''
        return cuts.volume(self.graph, comm, weight='weight')

    def calc_2dSE(self):
        '''
        get the 2D SE of the graph
        '''
        SE = 0
        for comm in self.division.values():
            g = self.get_cut(comm)
            v = self.get_volume(comm)
            SE += - (g / self.vol) * math.log2(v / self.vol)
            for node in comm:
                d = self.graph.degree(node, weight='weight')
                SE += - (d / self.vol) * math.log2(d / v)
        return SE


    def print_graph(self):
        fig, ax = plt.subplots()
        nx.draw(self.graph, ax=ax, with_labels=True)
        plt.show()

    def update_struc_data(self):
        '''
        calculate the volume, cut, communitiy mode SE, and leaf nodes SE of each cummunity,
        then store them into self.struc_data
        '''
        self.struc_data = {}  # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE]，... }
        for vname in self.division.keys():
            comm = self.division[vname]
            volume = self.get_volume(comm)
            cut = self.get_cut(comm)
            if volume == 0:
                vSE = 0
            else:
                vSE = - (cut / self.vol) * math.log2(volume / self.vol)
            vnodeSE = 0
            for node in comm:
                d = self.graph.degree(node, weight='weight')
                if d != 0:
                    vnodeSE -= (d / self.vol) * math.log2(d / volume)
            self.struc_data[vname] = [volume, cut, vSE, vnodeSE]

    def update_struc_data_2d(self):
        '''
        calculate the volume, cut, communitiy mode SE, and leaf nodes SE after merging each pair of cummunities,
        then store them into self.struc_data_2d
        '''
        self.struc_data_2d = {}  # {(comm1, comm2): [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], (comm1, comm3): [], ...}
        comm_num = len(self.division)
        for i in range(comm_num):
            for j in range(i + 1, comm_num):
                v1 = list(self.division.keys())[i]
                v2 = list(self.division.keys())[j]
                if v1 < v2:
                    k = (v1, v2)
                else:
                    k = (v2, v1)

                comm_merged = self.division[v1] + self.division[v2]
                gm = self.get_cut(comm_merged)
                vm = self.struc_data[v1][0] + self.struc_data[v2][0]
                if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
                    vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
                    vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
                else:
                    vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
                    vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0] / self.vol) * math.log2(
                        self.struc_data[v1][0] / vm) + \
                               self.struc_data[v2][3] - (self.struc_data[v2][0] / self.vol) * math.log2(
                        self.struc_data[v2][0] / vm)
                self.struc_data_2d[k] = [vm, gm, vmSE, vmnodeSE]


    def update_division_MinSE(self):
        '''
        greedily update the encoding tree to minimize 2D SE
        '''

        def Mg_operator(v1, v2):
            '''
            MERGE operator. It calculates the delta SE caused by mergeing communities v1 and v2,
            without actually merging them, i.e., the encoding tree won't be changed
            '''
            v1SE = self.struc_data[v1][2]
            v1nodeSE = self.struc_data[v1][3]

            v2SE = self.struc_data[v2][2]
            v2nodeSE = self.struc_data[v2][3]

            if v1 < v2:
                k = (v1, v2)
            else:
                k = (v2, v1)
            vm, gm, vmSE, vmnodeSE = self.struc_data_2d[k]
            delta_SE = vmSE + vmnodeSE - (v1SE + v1nodeSE + v2SE + v2nodeSE)  # lp-------why?
            return delta_SE

        # continue merging any two communities that can cause the largest decrease in SE,
        # until the SE can't be further reduced
        while True:
            comm_num = len(self.division)
            delta_SE = 99999
            vm1 = None
            vm2 = None
            for i in range(comm_num):
                for j in range(i + 1, comm_num):
                    v1 = list(self.division.keys())[i]
                    v2 = list(self.division.keys())[j]
                    new_delta_SE = Mg_operator(v1, v2)
                    if new_delta_SE < delta_SE:
                        delta_SE = new_delta_SE
                        vm1 = v1
                        vm2 = v2

            if delta_SE < 0:
                # Merge v2 into v1, and update the encoding tree accordingly
                for node in self.division[vm2]:
                    self.graph.nodes[node]['comm'] = vm1
                self.division[vm1] += self.division[vm2]
                self.division.pop(vm2)

                volume = self.struc_data[vm1][0] + self.struc_data[vm2][0]
                cut = self.get_cut(self.division[vm1])
                vmSE = - (cut / self.vol) * math.log2(volume / self.vol)
                vmnodeSE = self.struc_data[vm1][3] - (self.struc_data[vm1][0] / self.vol) * math.log2(
                    self.struc_data[vm1][0] / volume) + \
                           self.struc_data[vm2][3] - (self.struc_data[vm2][0] / self.vol) * math.log2(
                    self.struc_data[vm2][0] / volume)
                self.struc_data[vm1] = [volume, cut, vmSE, vmnodeSE]
                self.struc_data.pop(vm2)

                struc_data_2d_new = {}
                for k in self.struc_data_2d.keys():
                    if k[0] == vm2 or k[1] == vm2:
                        continue
                    elif k[0] == vm1 or k[1] == vm1:
                        v1 = k[0]
                        v2 = k[1]
                        comm_merged = self.division[v1] + self.division[v2]
                        gm = self.get_cut(comm_merged)
                        vm = self.struc_data[v1][0] + self.struc_data[v2][0]
                        if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
                            vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
                            vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
                        else:
                            vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
                            vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0] / self.vol) * math.log2(
                                self.struc_data[v1][0] / vm) + \
                                       self.struc_data[v2][3] - (self.struc_data[v2][0] / self.vol) * math.log2(
                                self.struc_data[v2][0] / vm)
                        struc_data_2d_new[k] = [vm, gm, vmSE, vmnodeSE]
                    else:
                        struc_data_2d_new[k] = self.struc_data_2d[k]
                self.struc_data_2d = struc_data_2d_new
            else:
                break

def hier_2D_SE_mini(edges, n_messages):
    '''
    hierarchical 2D SE minimization
    '''
    g = nx.Graph()
    g.add_weighted_edges_from(edges)
    seg = SE(g)
    seg.division = {j: [cluster] for j, cluster in enumerate(range(n_messages))}
    for k in seg.division.keys():
        for node in seg.division[k]:
            seg.graph.nodes[node]['comm'] = k
    seg.update_struc_data()
    seg.update_struc_data_2d()
    seg.update_division_MinSE()

    clusters = list(seg.division.values())

    return clusters

def print_graph(edges):
    g = nx.Graph()
    g.add_weighted_edges_from(edges)
    pos = nx.spring_layout(g,k=1)
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in g.edges(data=True)}
    nx.draw(g, pos, with_labels=True, node_size=1000, node_color='skyblue')  # 绘制节点
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,label_pos=0.5)
    plt.show()


def Similarity_calculation(feats,show = False):

    edges=[]
    num = len(feats[:,0])
    matrix = np.zeros((num, num))
    for i in range(num):
        a = torch.tensor(feats[i])
        for j in range(i+1, num):
            b = torch.tensor(feats[j])
            cosine_sim = torch.sum(cosine_similarity(a, b, dim=1)).item()/b.size(0)
            if cosine_sim >0:
                matrix[i,j] = cosine_sim
                matrix[j,i] = cosine_sim
                edges.append((i, j, cosine_sim))
    # print_graph(edges)

    clusters = hier_2D_SE_mini(edges, num)

    new_e = {}
    for cluster in clusters:
        for s in cluster:
            comm = cluster.copy()
            comm.remove(s)
            dic = {s:1}
            for c in comm:
                dic[c] = matrix[s , c ] if matrix[s , c ] > 0 else 0

            values = np.array(list(dic.values()))
            softmax_values = np.exp(values) / np.sum(np.exp(values))
            for i, key in enumerate(dic):
                dic[key] = softmax_values[i]

            new_e[s]=dic

    if show :
        g_e = []
        for cluster in clusters:
            result = list(itertools.combinations(cluster, 2))
            e_list = [(e[0],e[1],matrix[e[0],e[1]]) for e in result if matrix[e[0],e[1]]>0]
            g_e += e_list

        print_graph(g_e)

    return  new_e





if __name__ == '__main__':
    # 生成随机张量
    random_tensor_np = np.random.rand(6, 500, 32)
    Similarity_calculation(random_tensor_np)