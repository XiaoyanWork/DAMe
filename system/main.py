#!/usr/bin/env python
import copy
import sys

from system.flcore.servers.serverlocal_temp import FedLocal_Temp

sys.path.append("/home/yuxiaoyan/paper/pfedsed/system")
sys.path.append("/home/yuxiaoyan/paper/pfedsed/")
import numpy as np
import torch
torch.cuda.set_device(0)
from system.flcore.servers.serverapple import APPLE
from system.flcore.servers.serverlocal import FedLocal
from system.flcore.servers.serversed_trigger import FedSed_Trigger

import argparse
import time
import warnings
import logging
from flcore.servers.serveravg import FedAvg
from system.flcore.servers.serverala import FedALA
from system.flcore.servers.serverditto import Ditto
from system.flcore.servers.serverperavg import PerAvg
from system.flcore.servers.serverprox import FedProx
from system.flcore.servers.serversed import FedSed
from system.flcore.servers.serversfl import FedSfl
from system.flcore.trainmodel.model import GAT

from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)
torch.cuda.manual_seed(0)
# numpy 随机数种子
np.random.seed(0)

def run(args):

    reporter = MemReporter()
    model_str = args.model

    algorithm_cudas_dict = {"Local": 4, "FedAvg": 5, "FedProx": 6, "FedALA": 1, "PerAvg": 3, "Ditto": 4, "FedSfl": 7, "APPLE" : 1}   # baseline

    # 根据名称赋值消融属性  Ne 没有结构熵  Nb 没有贝叶斯  Nl没有成对损失
    # algorithm_cudas_dict = {"FedLocal_Temp" : 1, "FedSed": 4, "FedSed_Ne": 5, "FedSed_Nb": 2, "FedSed_Nl": 6, "FedSed_Trigger": 0, "FedSed_Model_Trigger" : 0, "FedSed_test" : 0, "FedSed_NL_test" : 0, "FedSed_NE_test" : 0, "FedSed_NB_test" : 0}  # 消融

    args.client_to_dataset_name = ["Arabic_Twitter", "China_Twitter", "English_Twitter", "French_Twitter", "Germany_Twitter", "Japan_Twitter", "Server_Twitter"]

    args.gpuid = algorithm_cudas_dict[args.algorithm]

    print(f"======================算法：{args.algorithm} 卡：{args.gpuid}=================")
    # 一共进行多少次实验
    for i in range(args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")


        # Generate 主干模型
        if model_str == "GAT":
            args.model = GAT(args.in_feat_dim, args.hidden_dim, args.out_dim, args.n_heads, True)
        else:
            raise NotImplementedError

        print(args.model)

        results = {}

        algo_dict = {}

        algo = args.algorithm
        # select algorithm
        if algo == "FedAvg":
            server = FedAvg(args, i)  # i代表低i-1次实验
        elif algo == "FedProx":
            server = FedProx(args, i)
        elif algo == "APPLE":
            server = APPLE(args, i)
        elif algo == "FedALA":
            server = FedALA(args, i)
        elif algo == "PerAvg":
            server = PerAvg(args, i)
        elif algo == "Ditto":
            server = Ditto(args, i)
        elif algo == "FedSfl":
            server = FedSfl(args, i)
        elif algo == "FedSed" or algo == "FedSed_Model_Trigger" or algo == "FedSed_Ne" or algo == "FedSed_Nb" or algo == "FedSed_Nl" or algo == "FedSed_Neb" or algo == "FedSed_Nel" or algo == "FedSed_Nbl" or algo == "FedSed_NL_test" or algo == "FedSed_NE_test" or algo == "FedSed_test" or algo == "FedSed_NB_test":
            server = FedSed(args, i)
        elif algo == "Local":
            server = FedLocal(args, i)
        elif algo == "FedSed_Trigger":
            server = FedSed_Trigger(args, i)
        else:
            raise NotImplementedError

        server.train()

        # 记录时间
        algo_dict["total_time"] = time.time() - total_start
        # 记录轮次
        algo_dict['eval_gap'] = args.eval_gap
        # 记录loss收敛
        algo_dict["nmis"] = server.nmi_dict

        # 记录聚合之後loss收敛
        algo_dict["nmis_later"] = server.nmi_dict_later
        # 记录聚合之後最终测试结果
        algo_dict["result_later"] = server.result_dict_later
        if algo == "FedSed_Trigger":
            algo_dict["result_dict_trigger_later"] = server.result_dict_trigger_later
        if algo == "FedSed":
            algo_dict["bays_weight"] = server.bays_weight

        results[algo] = algo_dict
        results["args"] = args

        print(results)

        # 将字典存储为npy文件
        np.save(f"results/{algo}.npy", results)

    print("All done!")

    reporter.report()

if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-use_cuda', "--use_cuda", type=bool, default=True,help="是否使用cuda")
    parser.add_argument('-gpuid', "--gpuid", type=str, default="0", help="cuda 的id")
    parser.add_argument('-m', "--model", type=str, default="GAT")
    parser.add_argument('-lr', "--lr", type=float, default=0.001, help="训练学习率")
    parser.add_argument('-gr', "--global_rounds", type=int, default=50)
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="每隔多少次验证一次")
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedSed_Trigger")  # "FedSed", "FedAvg", "FedProx", "FedALA", "PerAvg", "Ditto", "FedSfl",...
    parser.add_argument('-nc', "--num_clients", type=int, default=6, help="客户端总数")
    parser.add_argument('-t', "--times", type=int, default=1, help="Running times")
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=10)
    parser.add_argument('-validation_percent', "--validation_percent", type=float, default=0.1, help="验证机占比")
    parser.add_argument('-test_percent', "--test_percent", type=float, default=0.2, help="测试集占比")
    parser.add_argument('-max_local_epochs', "--max_local_epochs", type=int, default=1, help="本地迭代时间")
    parser.add_argument('-batch_size', "--batch_size", type=int, default=2000, help="批次大小")
    parser.add_argument('-agg_nums', "--agg_nums", type=int, default=6, help="聚合的客户端个数")

    # graph
    parser.add_argument('-residual', "--residual", type=bool, default=True, help="是否使用残差连接")
    parser.add_argument('-layer_norm', "--layer_norm", type=bool, default=True, help="是否使用layer")
    parser.add_argument('-batch_norm', "--batch_norm", type=bool, default=True, help="是否使用batch")
    parser.add_argument('-dropout', "--dropout", type=float, default=0.2, help="dropout")
    parser.add_argument('-n_heads', "--n_heads", type=int, default=4, help="多头注意力机制头数量")
    parser.add_argument('-out_dim', "--out_dim", type=int, default=32, help="输出维度")
    parser.add_argument('-in_feat_dim', "--in_feat_dim", type=int, default=386, help="输入维度")
    parser.add_argument('-hidden_dim', "--hidden_dim", type=int, default=48, help="隐藏层维度")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')

    # FedProx, PerAvg
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)

    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=60)
    parser.add_argument('-p', "--layer_idx", type=int, default=4, help="More fine-graind than its original paper.")
    parser.add_argument('-bt', "--beta", type=float, default=0.0)

    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1, help="本地私有化定制模型本地训练轮次")

    # FedSfl
    parser.add_argument('--agg', type=str, default='graph_v3', help='averaging strategy')
    parser.add_argument('--subgraph_size', type=int, default=30, help='k')
    parser.add_argument('--adjbeta', type=float, default=0.05, help='update ratio')
    parser.add_argument('--serveralpha', type=float, default=1, help='server prop alpha')
    parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
    parser.add_argument('--adjalpha', type=float, default=3, help='adj alpha')
    parser.add_argument('--gc_epoch', type=int, default=10, help='')
    parser.add_argument('--layers', type=int, default=2, help='聚合层数')

    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)

    # fedsed
    parser.add_argument('-server_data_path', "--server_data_path", type=str, default="../data_test/server", help="服务器端数据集路径")
    parser.add_argument('-server_sample', "--server_sample", type=float, default=0.5, help="服务器端数据集采样百分比")
    parser.add_argument("--shot", type=int, default=5, help="number of shot for few-shot learning")
    parser.add_argument("--split", type=str, default="val", help="split of dataset to evaluate")
    parser.add_argument("--output_dir", type=str, default="ceval_output", help="output directory")
    parser.add_argument("--save_best", type=str, default="results", help="directory to save the best model")
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="results2")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--performance_target", type=float, default=0.6)
    parser.add_argument("--pair_loss", type=bool, default=True, help="是否使用事件成對損失")
    parser.add_argument("--bays", type=bool, default=True, help="是否使用bays本地融合")
    parser.add_argument("--structural_entropy", type=bool, default=True, help="是否使用结构熵全局融合")
    parser.add_argument("--num_nodes", type=int, default=5000, help="生成随机图的结点个数")
    parser.add_argument("--edge_prob", type=float, default=0.6, help="生成边")
    parser.add_argument("--poison_rate", type=float, default=0.2, help="有毒数据因子")
    parser.add_argument("--trigger", type=bool, default=False, help="是否数据投毒")
    parser.add_argument("--model_trigger", type=bool, default=False, help="是否模型投毒")
    parser.add_argument("--lambdas", type=bool, default=False, help="是否记录lambda")

    args = parser.parse_args()

    # 规整bays、pair_loss、structural_entropy参数以进行更好的卡选择  {"FedSed": 1, "FedSed_Ne": 2, "FedSed_Nb": 3, "FedSed_Nl": 4, "FedSed_Neb": 5, "FedSed_Nel": 6, "FedSed_Nbl": 7}  # 消融
    if args.algorithm == "FedSed_Ne" or args.algorithm == "FedSed_NE_test":
        args.structural_entropy = False
    elif args.algorithm == "FedSed_Nb" or args.algorithm == "FedSed_NB_test":
        args.bays = False
    elif args.algorithm == "FedSed_Nl" or args.algorithm == "FedSed_NL_test":
        args.pair_loss = False
    elif args.algorithm == "FedSed_Neb":
        args.bays = False
        args.structural_entropy = False
    elif args.algorithm == "FedSed_Nel":
        args.pair_loss = False
        args.structural_entropy = False
    elif args.algorithm == "FedSed_Nbl":
        args.bays = False
        args.pair_loss = False

    # 是否投毒
    if args.algorithm == "FedSed_Trigger":
        args.trigger = True
    else:
        args.trigger = False

    if args.algorithm == "FedSed":
        args.lambdas = True
    else:
        args.lambdas = False

    # 是否进行模型投毒
    if args.algorithm == "FedSed_Model_Trigger":
        args.model_trigger = True
    else:
        args.model_trigger = False

    if args.use_cuda == True and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.use_cuda = False

    print("=" * 50)

    run(args)

