#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     d-dash.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2020-05-15
#
# @brief Baseline (simplied) implementation of D-DASH [1], a framework that
#        combines deep learning and reinforcement learning techniques to
#        optimize the quality of experience (QoE) of DASH, where the
#        policy-network is implemented based on feedforward neural network
#        (FNN) but without the target network and the replay memory.
#        The current implementation is based on PyTorch reinforcement learning
#        (DQN) tutorial [2].
#
# @remark [1] M. Gadaleta, F. Chiariotti, M. Rossi, and A. Zanella, “D-dash: A
#         deep Q-learning framework for DASH video streaming,” IEEE Trans. on
#         Cogn. Commun. Netw., vol. 3, no. 4, pp. 703–718, Dec. 2017.
#         [2] PyTorch reinforcement (DQN) tutorial. Available online:
#         https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


# import copy                     # for target network
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass
import torch.nn as nn
import torch
import copy

# global variables
# - DQL
CH_HISTORY = 1  # number of channel capacity history samples
BATCH_SIZE = 1000
EPS_START = 0.8
EPS_END = 0.0
EPS_DECAY = 200
LEARNING_RATE = 1e-4
# - FFN
N_I = 3 + CH_HISTORY  # input dimension (= state dimension)
N_H1 = 128
N_H2 = 256
N_O = 4
# - D-DASH
BETA = 2
GAMMA = 50
DELTA = 0.001
B_MAX = 20
B_THR = 10
T = 2  # segment duration
TARGET_UPDATE = 20
LAMBDA = 0.9
# RNN parameters

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
plt.ion()  # turn interactive mode on

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define neural network
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # define LSTM class
            input_size=N_I,  # 图片每行的数据像素点
            hidden_size=8,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.fc = nn.Linear(8, 4)  # 输出层

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        # out = self.out(r_out[:, -1, :])
        out = self.fc(r_out)
        return out


@dataclass
class State:
    """
    $s_t = (q_{t-1}, F_{t-1}(q_{t-1}), B_t, \bm{C}_t)$, which is a modified
    version of the state defined in [1].
    """

    sg_quality: int
    sg_size: float
    buffer: float
    ch_history: np.ndarray

    def tensor(self):
        return torch.tensor(
            np.concatenate(
                (
                    np.array([
                        self.sg_quality,
                        self.sg_size,
                        self.buffer]),
                    self.ch_history
                ),
                axis=None
            ),
            dtype=torch.float32
        )


@dataclass
class Experience:
    """$e_t = (s_t, q_t, r_t, s_{t+1})$ in [1]"""

    state: State
    action: int
    reward: float
    next_state: State


class ReplayMemory(object):
    """Replay memory based on a circular buffer (with overlapping)"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * self.capacity
        self.position = 0
        self.num_elements = 0

    def push(self, experience):
        # if len(self.memory) < self.capacity:
        #     self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        if self.num_elements < self.capacity:
            self.num_elements += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_num_elements(self):
        return self.num_elements


class ActionSelector(object):
    """
    Select an action based on the exploration policy.
    """

    def __init__(self, num_actions):
        self.steps_done = 0
        self.num_actions = num_actions

    def reset(self):
        self.steps_done = 0

    def increse_step_number(self):
        self.steps_done += 1

    def action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        # self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                net_output = policy_net_lstm(state.tensor().view(-1, 1, N_I))
                return int(torch.argmax(net_output[:, -1, :]))
        else:
            return random.randrange(self.num_actions)


# policy-network based on FNN with 2 hidden layers
policy_net = torch.nn.Sequential(
    torch.nn.Linear(N_I, N_H1),
    torch.nn.ReLU(),
    torch.nn.Linear(N_H1, N_H2),
    torch.nn.ReLU(),
    torch.nn.Linear(N_H2, N_O),
    torch.nn.Sigmoid()
).to(device)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

# policy_network based on RNN with 1 hidden layer
policy_net_lstm = RNN()
optimizer_lstm = torch.optim.Adam(policy_net_lstm.parameters(), lr=LEARNING_RATE)

# TODO: Implement target network
target_net = copy.deepcopy(policy_net_lstm)
target_net.load_state_dict(policy_net_lstm.state_dict())
target_net.eval()


def simulate_dash(sss, bws):
    # initialize parameters
    num_segments = sss.shape[0]  # number of segments
    num_qualities = sss.shape[1]  # number of quality levels

    # initialize replay memory and action_selector
    memory = ReplayMemory(1000)
    selector = ActionSelector(num_qualities)

    ##########
    # training
    ##########
    num_episodes = 30
    mean_sqs = np.empty(num_episodes)  # mean segment qualities
    mean_rewards = np.empty(num_episodes)  # mean rewards

    for i_episode in range(num_episodes):

        # TODO: use different video traces per episode

        sqs = np.empty(num_segments - CH_HISTORY)
        rewards = np.empty(num_segments - CH_HISTORY)
        mse_loss = torch.nn.MSELoss(reduction='mean')

        # initialize the state
        sg_quality = random.randrange(num_qualities)  # random action
        state = State(
            sg_quality=sg_quality,
            sg_size=sss[CH_HISTORY - 1, sg_quality],
            buffer=T,
            ch_history=bws[0:CH_HISTORY]
        )

        for t in range(CH_HISTORY, num_segments):
            sg_quality = selector.action(state)
            sqs[t - CH_HISTORY] = sg_quality

            # update the state
            tau = sss[t, sg_quality] / bws[t]
            buffer_next = T - max(0, state.buffer - tau)
            next_state = State(
                sg_quality=sg_quality,
                sg_size=sss[t, sg_quality],
                buffer=buffer_next,
                ch_history=bws[t - CH_HISTORY + 1:t + 1]
            )

            # calculate reward (i.e., (4) in [1]).
            downloading_time = next_state.sg_size / next_state.ch_history[-1]
            rebuffering = max(0, downloading_time - state.buffer)
            rewards[t - CH_HISTORY] = next_state.sg_quality \
                                      - BETA * abs(next_state.sg_quality - state.sg_quality) \
                                      - GAMMA * rebuffering - DELTA * max(0, B_THR - next_state.buffer) ** 2

            # store the experience in the replay memory
            experience = Experience(
                state=state,
                action=sg_quality,
                reward=rewards[t - CH_HISTORY],
                next_state=next_state
            )
            memory.push(experience)

            # move to the next state
            state = next_state

            #############################
            # optimize the policy network
            #############################
            if memory.get_num_elements() < BATCH_SIZE:
                continue
            experiences = memory.sample(BATCH_SIZE)
            state_batch = torch.stack([experiences[i].state.tensor()
                                       for i in range(BATCH_SIZE)])
            next_state_batch = torch.stack([experiences[i].next_state.tensor()
                                            for i in range(BATCH_SIZE)])
            action_batch = torch.tensor([experiences[i].action
                                         for i in range(BATCH_SIZE)])
            reward_batch = torch.tensor([experiences[i].reward
                                         for i in range(BATCH_SIZE)])

            # $Q(s_t, q_t|\bm{w}_t)$ in (13) in [1]
            # 1. policy_net generates a batch of Q(...) for all q values.
            # 2. columns of actions taken are selected using 'action_batch'.
            # state_action_values = policy_net_lstm(state_batch.view(-1, BATCH_SIZE, 5))

            state_Q_values = torch.squeeze(policy_net_lstm(state_batch.view(-1, BATCH_SIZE, N_I)))
            state_action_values = state_Q_values.gather(1, action_batch.view(BATCH_SIZE, -1))

            # $\max_{q}\hat{Q}(s_{t+1},q|\bar{\bm{w}}_t$ in (13) in [1]
            # TODO: Replace policy_net with target_net.
            target_values = torch.squeeze(target_net(next_state_batch.view(-1, BATCH_SIZE, N_I)))
            next_state_values = target_values.max(1)[0].detach()

            # expected Q values
            expected_state_action_values = reward_batch + (LAMBDA * next_state_values)

            # loss function, i.e., (14) in [1]
            loss = mse_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

            # optimize the model
            optimizer_lstm.zero_grad()
            loss.backward()
            for param in policy_net_lstm.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer_lstm.step()

            # TODO: Implement target network
            # # update the target network
            if t % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net_lstm.state_dict())

        # processing after each episode
        selector.increse_step_number()
        mean_sqs[i_episode] = sqs.mean()
        mean_rewards[i_episode] = rewards.mean()
        print("Mean Segment Quality[{0:2d}]: {1:E}".format(i_episode, mean_sqs[i_episode]))
        print("Mean Reward[{0:2d}]: {1:E}".format(i_episode, mean_rewards[i_episode]))

    return (mean_sqs, mean_rewards)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-V",
        "--video_trace",
        help="video trace file name; default is 'bigbuckbunny.npy'",
        default='bigbuckbunny.npy',
        type=str)
    parser.add_argument(
        "-C",
        "--channel_bandwidths",
        help="channel bandwidths file name; default is 'bandwidths.npy'",
        default='bandwidths.npy',
        type=str)
    args = parser.parse_args()
    video_trace = args.video_trace
    channel_bandwidths = args.channel_bandwidths

    # read data
    sss = np.load(video_trace)  # segment sizes [bit]
    bws = np.load(channel_bandwidths)  # channel bandwdiths [bit/s]

    # simulate D-DASH
    mean_sqs, mean_rewards = simulate_dash(sss, bws)

    # plot results
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(mean_rewards)
    axs[0].set_ylabel("Reward")
    axs[1].plot(mean_sqs)
    axs[1].set_ylabel("Video Quality")
    axs[1].set_xlabel("Video Episode")
    plt.show()
    input("Press ENTER to continue...")
    plt.close('all')
