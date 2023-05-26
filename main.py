import random

import gym
import numpy as np
import torch
import torch.nn as nn

from agent import Agent

env = gym.make("CartPole-v1")
s = env.reset()

EPSILON_DECAY = 50000
EPSILON_START = 1.0
EPSILON_END = 0.01
TARGET_UNDATE_FREQ = 10
n_episode = 5000
n_time = 1000
REWARD_BUFFER = np.empty(shape=n_episode)

n_state = len(s)
n_action = env.action_space.n
agent = Agent(n_state, n_action)
for episode_i in range(n_episode):
    episilon_reward = 0
    for step_i in range(n_time):
        epsilon = np.interp(episode_i * n_time + step_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        ran_sample = random.random()

        if ran_sample <= epsilon:
            a = env.action_space.sample()
        else:
            a = agent.online_net.act(s)  # TODO

        s_prime, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        agent.memo.add_memo(s, a, r, done, s_prime)
        s = s_prime
        episilon_reward += r

        if done:
            s = env.reset()
            REWARD_BUFFER[episode_i] = episilon_reward
            break
        batchs, batcha, batchr, batchdone, batchs_prime = agent.memo.sample()

        target_q_values = agent.target_net(batchs_prime)
        max_q_value = target_q_values.max(dim=1, keepdim=True)[0]  # get the largest q
        targets = batchr + agent.GAMMA * (1 - batchdone) * max_q_value

        # compute q_values
        q_value = agent.online_net(batchs)
        a_q_value = torch.gather(input=q_value, dim=1, index=batcha)

        # compute loss
        loss = nn.functional.smooth_l1_loss(targets, a_q_value)
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()  # grad descent
    if episode_i % TARGET_UNDATE_FREQ == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())

        print("Episode :{}".format(episode_i))
        print("avg reward:{}".format(np.mean(REWARD_BUFFER[:episode_i])))
