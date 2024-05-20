import time

from system.flcore.clients.clientapple import clientAPPLE
from system.flcore.clients.clientavg import clientAVG
from system.flcore.servers.serverbase import Server
import torch

class APPLE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientAPPLE)
        self.Budget = []

        self.client_models = [c.model_c for c in self.clients]

        train_samples = 0
        for client in self.clients:
            train_samples += len(client.train_indices)
        p0 = [len(client.train_indices) / train_samples for client in self.clients]

        for c in self.clients:
            c.p0 = p0


    def train(self):
        for i in range(self.global_rounds + 2):
            print(f"\n-------------Training Round number: {i}-------------")
            s_t = time.time()
            self.send_models()

            self.evaluate(i, "later")

            if i == self.global_rounds + 1:
                break   # 評估完成後退出

            for client in self.clients:
                client.train(i)

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

    def send_models(self):
        assert (len(self.clients) > 0)

        self.client_models = [c.model_c for c in self.clients]
        for client in self.clients:

            client.set_models(self.client_models)

