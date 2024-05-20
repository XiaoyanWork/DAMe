import copy
import os

from sklearn.cluster import KMeans
from sklearn import metrics

from system.utils.ceval_bayseianOPT_1 import extract_embeddings
from system.utils.data_utils import evaluate
from matplotlib import gridspec
os.environ['CUDA_VISIBLE_DEVICES'] = '0,8'
from tqdm import tqdm
import json
import torch
import numpy as np
import pandas as pd
import gc
import argparse

import time
import random
import dgl
import networkx as nx
# from skopt.space import Real
# from skopt import Optimizer
# from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi
from skopt.space import Real
from skopt import Optimizer
from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi
from bayes_opt import UtilityFunction
import matplotlib.pyplot as plt
def bays_validation(g, model, args, validation_indices):
    extract_features, extract_labels = extract_embeddings(g, model, validation_indices, args)

    nmi, ami, ari, nmi_hdbcan, ami_hdbcan, ari_hdbcan = evaluate(extract_features, extract_labels, np.arange(0, len(validation_indices), 1),
                                                                 0, "")

    return nmi


def calculate_merge_weight(score1, score2):
    total = score1 + score2
    return score1 / total, score2 / total


def merge_models(model1, model2, weight1):
    weight2 = 1 - weight1
    scale_factor = weight2 / weight1

    # fused model = w1 x model1_param + w2 x model2_param
    intermediate_state_dict = {
        key: model1.state_dict()[key] + model2.state_dict()[key] * scale_factor
        for key in model1.state_dict()
    }
    for key in intermediate_state_dict:
        intermediate_state_dict[key] = intermediate_state_dict[key] * weight1

    del model1, model2
    torch.cuda.empty_cache()

    return intermediate_state_dict  # 返回融合模型的存储地址


def update_gp_model(iteration_records):
    space = [Real(0.6, 1.0, name="weight1")]
    optimizer = Optimizer(dimensions=space, random_state=42)
    for record in iteration_records:
        optimizer.tell([record["weight1"]], -record["nmi"])
    next_weight = optimizer.ask()
    return next_weight, optimizer


def calculate_annealing_factor(iteration, max_iterations, initial_temp=1.0):
    return initial_temp * (1 - iteration / (max_iterations + 1))


def calculate_step_size(step_size, nmi, best_nmi, performance_target, iteration, max_iterations):
    if nmi > best_nmi:
        step_size *= 0.9
    else:
        distance_to_target = max(performance_target - nmi, 0)
        annealing_factor = calculate_annealing_factor(iteration, max_iterations)
        if distance_to_target > 0 and random.random() < simulated_annealing(iteration, max_iterations):
            step_size *= (1 + distance_to_target * annealing_factor)
        else:
            step_size *= 1.1
    return step_size


def simulated_annealing(iteration, max_iterations, initial_temp=1.0):
    # Simple simulated annealing function based on iteration count
    temp = calculate_annealing_factor(iteration, max_iterations, initial_temp)
    return np.exp(-1.0 / temp)


def posterior(optimizer, x_obs, y_obs, grid):
    if not optimizer.models:
        raise RuntimeError("No models found. Ensure that the optimizer has been told at least one observation.")

    model = optimizer.models[-1]
    mu, sigma = model.predict(grid, return_std=True)
    return mu, sigma

def select_next_weight_based_on_metric(optimizer, metric='ei'):
    x_values = np.linspace(0.6, 1.0, 100).reshape(-1, 1)
    mu, sigma, ei_values, lcb_values = calculate_decision_metrics(optimizer, x_values)

    if metric == 'ei':
        next_index = np.argmax(ei_values)  # 最大预测结果的索引作为下一个观察点的索引
    elif metric == 'lcb':
        next_index = np.argmin(lcb_values)
    elif metric == 'mu':
        next_index = np.argmax(mu)
    elif metric == 'sigma':
        next_index = np.argmax(sigma)

    next_weight = x_values[next_index][0]  # 该索引对应的x值即为下一步的权重 λ
    return [next_weight]

def calculate_decision_metrics(optimizer, x_values):
    if not optimizer.models:  # 检查优化器对象中是否存在模型
        raise RuntimeError("No models found. Ensure that the optimizer has been told at least one observation.")

    model = optimizer.models[-1]  # return -> list: Regression models used to fit observations and compute acquisition function.
    mu, sigma = model.predict(x_values,
                              return_std=True)  # base_estimator,  default: "GP"; gaussian process; returns std(Y | x) along with E[Y | x].

    # Function to minimize over the posterior distribution.
    ei_values = gaussian_ei(x_values, model)
    lcb_values = gaussian_lcb(x_values, model)

    return mu, sigma, ei_values, lcb_values

def init_observation(best_model, second_best_model, validation_indices, args, g):
    device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
    merge_ratio_range = np.random.choice(np.linspace(0.6, 1.0, 40), size=5, replace=False)
    ori_model = copy.deepcopy(best_model)
    initial_x_iters = []
    initial_func_vals = []
    for weight in merge_ratio_range:
        merged_model_dict = merge_models(best_model, second_best_model, weight)
        ori_model.load_state_dict(merged_model_dict)
        ori_model = ori_model.to(device)
        nmi = bays_validation(g, ori_model, args, validation_indices[:1000])
        initial_x_iters.append([weight])
        initial_func_vals.append(-nmi)
    # 删除不再需要的参数
    del ori_model, merged_model_dict
    return initial_x_iters, initial_func_vals

def plot_gp(optimizer, x, y, plot_path):
    if not optimizer.models:
        raise RuntimeError("No models found. Ensure that the optimizer has been told at least one observation.")

    model = optimizer.models[-1]

    fig = plt.figure(figsize=(16, 10))
    #steps = len(optimizer.space)
    steps = optimizer.space.n_dims
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size':30}
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    #x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    #y_obs = np.array([res["target"] for res in optimizer.res])
    x_obs = np.array(optimizer.Xi)
    y_obs = np.array(optimizer.yi)

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((0.9, 1))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})

    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, model, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((0.6, 1))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    plt.savefig(plot_path)
    plt.close()

def iterative_adjustment(models, g, args, validation_indices, filename):
    device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
    ori_model = copy.deepcopy(models[0])
    start_time = time.time()
    best_score = -np.inf
    best_weights = None
    iteration_records = []

    # 细节参考这个文档: https://scikit-optimize.github.io/stable/modules/generated/skopt.optimizer.Optimizer.html#skopt.optimizer.Optimizer
    optimizer = Optimizer(dimensions=[Real(0.6, 1.0, name="weight1")], n_initial_points=5, random_state=42)

    # get model init scores
    model_scores = []
    model_scores.append(bays_validation(g, models[0], args, validation_indices[:1000]))
    model_scores.append(bays_validation(g, models[1], args, validation_indices[:1000]))

    # get best model
    best_model_idx = np.argmax(model_scores)
    second_best_model_idx = 1 - best_model_idx
    best_model = models[best_model_idx]
    second_best_model = models[second_best_model_idx]

    step_size = 0.04
    last_nmi = 0
    merge_ratio_range = np.linspace(0.6, 1.0, 100)

    # 获得初始值（先验
    # initial_x_iters = [[0.9186171947440932], [0.6733739159464656], [0.9118764001091078], [0.8387400631785948], [0.7783331011414365]]
    # initial_func_vals = [-0.5365905701338546, -0.5589774157054157, -0.538634738821411, -0.5582477551157603, -0.5606133468676191]
    initial_x_iters, initial_func_vals = init_observation(best_model, second_best_model, validation_indices, args, g)
    print()
    for x, y in zip(initial_x_iters, initial_func_vals):
        optimizer.tell(x, y)  # Record an observation (or several) of the objective function.

    # 开始贝叶斯优化
    for iteration in range(6, args.max_iterations + 1):
        metric = 'ei'
        suggested_weight = select_next_weight_based_on_metric(optimizer, metric=metric)
        # suggested_weight = optimizer.ask()
        if isinstance(suggested_weight, list) and len(suggested_weight) > 0:
            weight1 = suggested_weight[0]
        weight2 = 1 - weight1

        merged_model_dict = merge_models(best_model, second_best_model, weight1)

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # special task 的评估指标，这里换成我们联邦的指标如NMI和其他什么的吧?
        ori_model.load_state_dict(merged_model_dict)
        ori_model = ori_model.to(device)

        nmi = bays_validation(g, ori_model, args, validation_indices)
        # nmi = bays_validation(g, ori_model, args, validation_indices[:1000])

        # Obtain current accuracy
        nmi_path = os.path.join(args.output_dir, "nmi.json")
        if os.path.exists(nmi_path):
            with open(nmi_path, "r") as f:
                nmis = json.load(f)
            nmi = np.mean(list(nmis.values()))
        # 获得在task上评估指标metric的得分acc，为实际观测值f(x)，加入观测点集合{x,f(x)}
        # =======================================================================

        optimizer.tell(suggested_weight, -nmi)  # 高斯拟合的预测拉姆达，对应的实际fx=f(x)值

        step_size = calculate_step_size(step_size, nmi, best_score, args.performance_target, iteration, args.max_iterations)

        iteration_records.append({"iteration": iteration, "weight1": weight1, "nmi": nmi})

        # Save detailed GP data
        gp_details = {
            'iteration': iteration,
            'tested_weight': weight1,
            'nmi': nmi,
            "ei": [],
            "lcb": [],
            'gp_predictions': [],
            'gp_uncertainties': []
        }

        if iteration > 5 and optimizer.models:
            model_gp = optimizer.models[-1]
            x_transformed = [[x] for x in merge_ratio_range.tolist()]
            mean, std = model_gp.predict(x_transformed, return_std=True)

            gp_details["gp_predictions"] = mean.tolist()
            gp_details["gp_uncertainties"] = std.tolist()
            ei_values = gaussian_ei(x_transformed, model_gp)
            lcb_values = gaussian_lcb(x_transformed, model_gp)
            gp_details["ei"] = ei_values.tolist()
            gp_details["lcb"] = lcb_values.tolist()

            # y = [model_gp.predict(x.reshape(1, -1), return_std=False)[0] for x in x_transformed]
            # y = [model_gp.predict(np.array(x).reshape(1, -1), return_std=False)[0] for x in x_transformed]
            y = [model_gp.predict(np.array(x).reshape(1, -1), return_std=False)[0] for x in x_transformed]
            plot_gp(optimizer, x_transformed, y, f'/home/yuxiaoyan/paper/pfedsed/system/results/plot/gp_iter1760-9_{filename}_{iteration}.png')

        # Append the GP details to iteration_records: 已观测点集合
        iteration_records.append(gp_details)

        if nmi > best_score:
            best_score = nmi
            best_weights = (weight1, 1 - weight1)
        else:
            if nmi < last_nmi and random.random() < simulated_annealing(iteration, args.max_iterations):
                step_size *= 1.1
            else:
                step_size *= 0.9

        last_nmi = nmi

        if iteration == args.max_iterations:
            break

        if iteration < args.max_iterations:
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    del ori_model
    return best_weights, best_score, iteration_records, optimizer, best_model

def getdata():
    # 生成随机图
    num_nodes = 500
    edge_prob = 0.6
    nx_graph = nx.generators.random_graphs.erdos_renyi_graph(num_nodes, edge_prob, directed=False, seed=0)
    g = dgl.from_networkx(nx_graph)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.readonly(readonly_state=True)
    g.ndata['features'] = torch.normal(mean=0, std=1, size=(num_nodes, 386))
    g.ndata['labels'] = torch.randint(low=0, high=2, size=(num_nodes,))
    train_indices = torch.randperm(num_nodes)

    return g, train_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shot", type=int, default=5, help="number of shot for few-shot learning"
    )
    parser.add_argument(
        "--split", type=str, default="val", help="split of dataset to evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="ceval_output", help="output directory"
    )
    parser.add_argument(
        "--save_best", type=str, default="results", help="directory to save the best model"
    )
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="results2")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--performance_target", type=float, default=0.6)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--gpuid", type=int, default=0)
    args = parser.parse_args()

    g, train_indices = getdata()

    model = GAT(386, 48, 32, 3, use_residual=True)
    global_model = GAT(386, 48, 32, 3, use_residual=True)

    best_weights, best_score, iteration_records, optimizer, best_model = iterative_adjustment((model, global_model), g, args, train_indices)
    print(best_weights)


