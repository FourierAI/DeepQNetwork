#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: d-dash_LSTM.py
# @time: 2020-06-20 16:19
# @desc:

# import copy                     # TASK2: for target network
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass

# global variables
# - DQL
CH_HISTORY = 1  # number of channel capacity history samples
# BATCH_SIZE = 1000
EPS_START = 0.8
EPS_END = 0.0
LEARNING_RATE = 1e-4
MEMORY_SIZE = 10000
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
# - LSTM
INPUT_SIZE = 3 + CH_HISTORY
HIDDEN_SIZE = 8
NUM_LAYERS = 1
OUPUT_SIZE = 4

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
plt.ion()  # turn interactive mode on

# set device
device = torch.device("cpu")


# Define neural network
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # define LSTM class
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,  # if batch_first : out = (batch, time_step, input_size)
        )

        self.fc = nn.Linear(HIDDEN_SIZE, OUPUT_SIZE)

    def forward(self, x):
        self.rnn.flatten_parameters()
        r_out, (h_n, h_c) = self.rnn(x, None)

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
    """Replay memory based on a list"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_num_elements(self):
        return len(self.memory)

    # """Replay memory based on a circular buffer (with overlapping)"""
    # def __init__(self, capacity):
    #     self.capacity = capacity
    #     self.memory = [None] * self.capacity
    #     self.position = 0
    #     self.num_elements = 0

    # def push(self, experience):
    #     # if len(self.memory) < self.capacity:
    #     #     self.memory.append(None)
    #     self.memory[self.position] = experience
    #     self.position = (self.position + 1) % self.capacity
    #     if self.num_elements < self.capacity:
    #         self.num_elements += 1

    # def sample(self, batch_size):
    #     return random.sample(self.memory, batch_size)

    # def get_num_elements(self):
    #     return self.num_elements


class ActionSelector(object):
    """
    Select an action based on the exploration policy.
    """

    def __init__(self, num_actions, num_segments, greedy_policy=False):
        self.steps_done = 0
        self.num_actions = num_actions
        self.num_segments = num_segments
        self.greedy_policy = greedy_policy

    def reset(self):
        self.steps_done = 0

    # def set_greedy_policy(self):
    #     self.greedy_policy = True

    def increse_step_number(self):
        self.steps_done += 1

    def action(self, state):
        if self.greedy_policy:
            with torch.no_grad():
                return int(torch.argmax(policy_net_lstm(state.tensor().view(-1, 1, INPUT_SIZE))))
        else:
            sample = random.random()
            x = 20 * (self.steps_done / self.num_segments) - 6.  # scaled s.t. -6 < x < 14
            eps_threshold = EPS_END + (EPS_START - EPS_END) / (1. + math.exp(x))
            # self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    return int(torch.argmax(policy_net_lstm(state.tensor().view(-1, 1, INPUT_SIZE))))
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

policy_net_lstm = RNN().to(device)
optimizer_lstm = torch.optim.Adam(policy_net_lstm.parameters(), lr=LEARNING_RATE)


# TASK2: Implement target network
# target_net = copy.deepcopy(policy_net)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()


def simulate_dash(sss, bws, memory, phase, batch_size):
    # initialize parameters
    num_segments = sss.shape[0]  # number of segments
    num_qualities = sss.shape[1]  # number of quality levels

    if phase == 'train':
        # initialize action_selector
        selector = ActionSelector(num_qualities, num_segments, greedy_policy=False)
    elif phase == 'test':
        selector = ActionSelector(num_qualities, num_segments, greedy_policy=True)
    else:
        sys.exit(phase + " is not supported.")

    ##########
    # training
    ##########
    num_episodes = 300
    mean_sqs = np.empty(num_episodes)  # mean segment qualities
    mean_rewards = np.empty(num_episodes)  # mean rewards
    for i_episode in range(num_episodes):
        sqs = np.empty(num_segments - CH_HISTORY)
        rewards = np.empty(num_segments - CH_HISTORY)

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
            if memory.get_num_elements() < batch_size:
                continue
            experiences = memory.sample(batch_size)
            state_batch = torch.stack([experiences[i].state.tensor()
                                       for i in range(batch_size)])
            next_state_batch = torch.stack([experiences[i].next_state.tensor()
                                            for i in range(batch_size)])
            action_batch = torch.tensor([experiences[i].action
                                         for i in range(batch_size)], dtype=torch.long)
            reward_batch = torch.tensor([experiences[i].reward
                                         for i in range(batch_size)], dtype=torch.float32)

            # $Q(s_t, q_t|\bm{w}_t)$ in (13) in [1]
            # 1. policy_net generates a batch of Q(...) for all q values.
            # 2. columns of actions taken are selected using 'action_batch'.
            state_Q_values = torch.squeeze(policy_net_lstm(state_batch.view(-1, batch_size, INPUT_SIZE)))
            state_action_values = state_Q_values.gather(1, action_batch.view(batch_size, -1))

            # $\max_{q}\hat{Q}(s_{t+1},q|\bar{\bm{w}}_t$ in (13) in [1]
            # TODO: Replace policy_net with target_net.
            target_values = torch.squeeze(
                policy_net_lstm(next_state_batch.view(-1, batch_size, INPUT_SIZE)))
            next_state_values = target_values.max(1)[0].detach()

            # expected Q values
            expected_state_action_values = reward_batch + (LAMBDA * next_state_values)

            # loss fuction, i.e., (14) in [1]
            mse_loss = torch.nn.MSELoss(reduction='mean')
            loss = mse_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

            # optimize the model
            optimizer_lstm.zero_grad()
            loss.backward()
            for param in policy_net_lstm.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer_lstm.step()

            # TASK2: Implement target network
            # # update the target network
            # if t % TARGET_UPDATE == 0:
            #     target_net.load_state_dict(policy_net.state_dict())

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
        "--train_video_trace",
        help="training video trace file name; default is 'bigbuckbunny.npy'",
        default='bigbuckbunny.npy',
        type=str)
    parser.add_argument(
        "--test_video_trace",
        help="testing video trace file name; default is 'bear.npy'",
        default='bear.npy',
        type=str)
    parser.add_argument(
        "-C",
        "--channel_bandwidths",
        help="channel bandwidths file name; default is 'bandwidths.npy'",
        default='bandwidths.npy',
        type=str)
    parser.add_argument(
        "-B",
        "--batch_size",
        help="batch size; default is 1000",
        default=1000,
        type=int)
    args = parser.parse_args()
    train_video_trace = args.train_video_trace
    test_video_trace = args.test_video_trace
    channel_bandwidths = args.channel_bandwidths
    batch_size = args.batch_size

    # initialize channel BWs and replay memory
    bws = np.load(channel_bandwidths)  # channel bandwdiths [bit/s]
    memory = ReplayMemory(MEMORY_SIZE)

    # training phase
    sss = np.load(train_video_trace)  # segment sizes [bit]
    train_mean_sqs, train_mean_rewards = simulate_dash(sss, bws, memory, 'train', batch_size)

    # testing phase
    sss = np.load(test_video_trace)  # segment sizes [bit]
    test_mean_sqs, test_mean_rewards = simulate_dash(sss, bws, memory, 'test', batch_size)

    # plot results
    mean_sqs = np.concatenate((train_mean_sqs, test_mean_sqs), axis=None)
    mean_rewards = np.concatenate((train_mean_rewards, test_mean_rewards), axis=None)
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(mean_rewards)
    axs[0].set_ylabel("Reward")
    axs[0].vlines(len(train_mean_rewards), *axs[0].get_ylim(), colors='red', linestyles='dotted')
    axs[1].plot(mean_sqs)
    axs[1].set_ylabel("Video Quality")
    axs[1].set_xlabel("Video Episode")
    axs[1].vlines(len(train_mean_rewards), *axs[1].get_ylim(), colors='red', linestyles='dotted')
    plt.savefig('d-dash_lstm.pdf', format='pdf')
    plt.show()
    # input("Press ENTER to continue...")
    plt.close('all')
