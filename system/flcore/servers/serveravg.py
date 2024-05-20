import time
from system.flcore.clients.clientavg import clientAVG
from system.flcore.servers.serverbase import Server
import torch

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientAVG)
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
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        self.send_models()
        # 開始驗證
        self.test("later")
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
