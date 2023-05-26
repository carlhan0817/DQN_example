import random

import numpy as np
import torch
import torch.nn as nn


class Replaymemory:
    def __init__(self, n_s, n_a):
        self.s = n_s
        self.a = n_a
        self.memorysize = 1000
        self.batchsize = 64

        self.all_s = np.empty(shape=(self.memorysize, self.s), dtype=np.float64)
        self.all_a = np.random.randint(0, self.a, self.memorysize, dtype=np.uint8)
        self.all_r = np.empty(self.memorysize, dtype=np.float32)
        self.all_done = np.random.randint(0, 2, self.memorysize, dtype=np.uint8)
        self.all_s_prime = np.empty(shape=(self.memorysize, self.s), dtype=np.float32)
        self.t_memo = 0
        self.t_max = 0

    def add_memo(self, s, a, r, done, s_prime):
        self.all_s[self.t_memo] = s
        self.all_a[self.t_memo] = a
        self.all_r[self.t_memo] = r
        self.all_done[self.t_memo] = done
        self.all_s_prime[self.t_memo] = s_prime
        self.t_max = max(self.t_max, self.t_memo + 1)
        self.t_memo = (self.t_memo + 1) % self.memorysize

    def sample(self):
        if self.t_max > self.batchsize:
            indexs = random.sample(range(self.t_max), self.batchsize)  # tmax > batchsize
        else:
            indexs = range(0, self.t_max)
        batch_s = []
        batch_a = []
        batch_r = []
        batch_done = []
        batch_s_prime = []

        for idx in indexs:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_done.append(self.all_done[idx])
            batch_s_prime.append(self.all_s_prime[idx])

            batch_s_tenser = torch.as_tenser(np.asarray(batch_s), dtype=torch.float32)
            batch_a_tenser = torch.as_tenser(np.asarray(batch_a), dtype=torch.uint8).unsqueeze(-1)
            batch_r_tenser = torch.as_tenser(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
            batch_done_tenser = torch.as_tenser(np.asarray(batch_done), dtype=torch.float32).unsqueeze(-1)
            batch_s_prime_tenser = torch.as_tenser(np.asarray(batch_s_prime), dtype=torch.float32)

        return batch_s_tenser, batch_a_tenser, batch_r_tenser, batch_done_tenser, batch_s_prime_tenser


class DQN(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=n_input, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=n_output)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_tenser = torch.as_tensor(obs, dtype=torch.float32)
        q_value = self(obs_tenser.unsqueeze(0))
        max_q_index = torch.argmax(q_value)
        action = max_q_index.detach().item()

        return action


class Agent:
    def __init__(self, n_input, n_output):
        self.input = n_input
        self.output = n_output

        self.GAMMA = 0.9
        self.learning_rate = 1e-3
        self.memo = Replaymemory(self.input, self.output)

        self.online_net = DQN(self.input, self.output)
        self.target_net = DQN(self.input, self.output)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
