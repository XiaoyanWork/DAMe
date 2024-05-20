import time
import torch
from system.flcore.clients.clientperavg import clientPerAvg
from system.flcore.servers.serverbase import Server


class PerAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientPerAvg)

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 2):
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
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)  # 记录每一次迭代的时间开销
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

        self.send_models()
        # 開始驗證
        self.test("later")
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))


    def evaluate(self, i, pos="before"):
        if i % self.eval_gap == 0:
            print(f"\n第{i}轮次时模型验证:")
            for client in self.clients:
                client.train_one_step()
                nmi, ami, ari, nmi_hdbcan, ami_hdbcan, ari_hdbcan, loss = client.validation()
                if pos == "before":
                    self.nmi_dict[self.client_to_dataset_name[client.id]].append(round(nmi, 2))
                else:
                    self.nmi_dict_later[self.client_to_dataset_name[client.id]].append(round(nmi, 2))

                print(f"Client {client.id} - {self.client_to_dataset_name[client.id]}  NMI: {nmi}, AMI: {ami}, ARI: {ari}, NMI_hdbcan: {nmi_hdbcan}, AMI_hdbcan: {ami_hdbcan}, ARI_hdbcan: {ari_hdbcan}")

