import torch
import copy
import time
from system.utils.dlg import DLG
import random

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = torch.device("cuda:{}".format(0) if args.use_cuda else "cpu")
        self.client_to_dataset_name = args.client_to_dataset_name
        self.global_rounds = args.global_rounds
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.algorithm = args.algorithm
        self.learning_rate_decay = args.learning_rate_decay

        self.clients = []
        self.select_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.times = times
        self.eval_gap = args.eval_gap

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.model_trigger = args.model_trigger  # 是否进行模型投毒
        # loss_dict
        self.nmi_dict = {self.client_to_dataset_name[index]: [] for index in range(args.num_clients)}  # 记录损失
        self.result_dict = {self.client_to_dataset_name[index]: [] for index in range(args.num_clients)}  # 记录测试结果
        self.nmi_dict_later = {self.client_to_dataset_name[index]: [] for index in range(args.num_clients)}  # 记录损失
        self.result_dict_later = {self.client_to_dataset_name[index]: [] for index in range(args.num_clients)}  # 记录测试结果

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            client = clientObj(self.args, id=i)
            self.clients.append(client)

    def evaluate(self, i, pos="before"):
        if i % self.eval_gap == 0:
            print(f"\n第{i}轮次时模型验证:")
            for client in self.clients:
                nmi, ami, ari, nmi_hdbcan, ami_hdbcan, ari_hdbcan, loss = client.validation()
                if pos == "before":
                    self.nmi_dict[self.client_to_dataset_name[client.id]].append(round(nmi, 2))
                else:
                    self.nmi_dict_later[self.client_to_dataset_name[client.id]].append(round(nmi, 2))

                print(f"Client {client.id} - {self.client_to_dataset_name[client.id]}  NMI: {nmi}, AMI: {ami}, ARI: {ari}, NMI_hdbcan: {nmi_hdbcan}, AMI_hdbcan: {ami_hdbcan}, ARI_hdbcan: {ari_hdbcan}")

    def test(self, pos="before"):
        print(f"\n最后一轮次时模型测试:")
        for client in self.clients:
            for time in range(5):

                nmi, ami, ari, nmi_hdbcan, ami_hdbcan, ari_hdbcan = client.test()
                if pos == "before":
                    self.result_dict[self.client_to_dataset_name[client.id]].append([round(nmi, 2), round(ami, 2),round(ari, 2)])
                else:
                    self.result_dict_later[self.client_to_dataset_name[client.id]].append([round(nmi, 2), round(ami, 2),round(ari, 2)])

                if time == 0:
                    print(f"Client {client.id} - {self.client_to_dataset_name[client.id]} NMI: {round(nmi, 2)}, AMI: {round(ami, 2)}, ARI: {round(ari, 2)}, NMI_hdbcan: {round(nmi_hdbcan, 2)}, AMI_hdbcan: {round(ami_hdbcan, 2)}, ARI_hdbcan: {round(ari_hdbcan, 2)}")

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_model)

    def receive_models(self):
        assert (len(self.clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        # 随机选择4个客户端进行聚合
        self.select_clients = random.sample(self.clients, self.args.agg_nums)
        for client in self.select_clients:
            tot_samples += client.data_len
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.data_len)
            if client.id == 2 and self.model_trigger: # 对英文客户端模型进行投毒
                self.uploaded_models.append(self.model_trigger_function(client.model))
            else:
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def model_trigger_function(self, model):
        with torch.no_grad():
            for param in model.parameters():
                # 生成与参数形状相同的随机噪声
                noise = torch.randn_like(param)
                noise = torch.clamp(noise, -0.1, 0.1)
                # 将噪声加到原始参数上
                param.add_(noise)

        # 返回修改后的模型
        return model

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

