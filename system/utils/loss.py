
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations
import torch.nn.functional as F


# 三元组损失
def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

# 获取三元组损失
class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    # 1、获取欧式距离矩阵
    # 2、建立锚点和正样本两两对（锚点进行for循环遍历）
    # 3、建立负样本
    # 4、建立三元组加入triplets
    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()  # seed_node_size * (out_dim * head)
        # 获取欧式距离矩阵
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()  # seed_node_size
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            # 是当前标签的index
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            # 不是当前标签的index
            negative_indices = np.where(np.logical_not(label_mask))[0]
            # 创建所有锚点-正样本对的两两组合
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)
            # 计算锚点-正样本对s之间的距离
            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    # 锚点 正例 负例
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:

            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                              negative_selection_fn=hardest_negative,
                                                                                              cpu=cpu)

def random_hard_negative(loss_values):
    #
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                             negative_selection_fn=random_hard_negative,
                                                                                             cpu=cpu)

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin=3.0, triplet_selector=RandomNegativeTripletSelector(3.0)):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target, get_triplet=False, triplets=None):
        if triplets == None:
            triplets = self.triplet_selector.get_triplets(embeddings, target)   # 抽样三元组

        if embeddings.is_cuda:
            triplets = triplets.to(embeddings.device)

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)  #
        if get_triplet:
            return losses.mean(), triplets.cpu()
        return losses.mean(), len(triplets)

# 成对损失
def add_pair_loss(pred, batch_labels):
    pairs, pair_labels, pair_matrix = pairwise_sample(batch_labels)
    pairs = pairs.cuda()
    pair_matrix = pair_matrix.cuda()
    pair_labels = pair_labels.unsqueeze(-1).cuda()

    pos_indices = torch.where(pair_labels > 0)
    neg_indices = torch.where(pair_labels == 0)
    neg_ind = torch.randint(0, neg_indices[0].shape[0], [5 * pos_indices[0].shape[0]]).cuda()
    neg_dis = (pred[pairs[neg_indices[0][neg_ind], 0]] - pred[pairs[neg_indices[0][neg_ind], 1]]).pow(2).sum(1).unsqueeze(-1)
    pos_dis = (pred[pairs[pos_indices[0], 0]] - pred[pairs[pos_indices[0], 1]]).pow(2).sum(1).unsqueeze(-1)
    pos_dis = torch.cat([pos_dis] * 5, 0)
    # 是否应该用每个正样本和每个负样本都减一遍
    pairs_indices = torch.where(torch.clamp(pos_dis + 16 - neg_dis, min=0.0) > 0)
    compare_loss = torch.mean(torch.clamp(pos_dis + 16 - neg_dis, min=0.0)[pairs_indices[0]])

    pred = F.normalize(pred, 2, 1)
    pair_out = torch.mm(pred, pred.t())
    pair_loss = (pair_matrix - pair_out).pow(2).mean()
    print(f"Batch: 成对损失:{compare_loss.item()},正交损失:{100 * pair_loss.item()},total loss:{(compare_loss + 100 * pair_loss).item()}")
    # 判断成对损失是否为nan
    if torch.isnan(compare_loss):
        loss = 100 * pair_loss
    else:
        loss = 100 * pair_loss + compare_loss
    return loss

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
