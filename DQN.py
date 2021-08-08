import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib
import random
from Storage import RolloutStorage
import copy

EPISILO = 0.9
Q_NETWORK_ITERATION = 1

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Base(nn.Module):

    def __init__(self, n_action, hidden_size):

        super(Base, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.f = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
                init_(nn.Linear(hidden_size, n_action)))

    def forward(self, x):
        
        return self.f(x)

class Net(nn.Module):
    def __init__(self, n_state, n_action, hidden_size = 128):
        super(Net, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.n_action = n_action

        self.actor = nn.Sequential(
            init_(nn.Linear(n_state, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.base = nn.ModuleList([Base(n_action[i], hidden_size) for i in range(len(n_action))])

    def forward(self,x):
        x = self.actor(x)
        return [self.base[i](x) for i in range(len(self.n_action))]

class DQN():
    def __init__(self, S_N, T_N, alpha, lr = 0.01, device = None):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = S_N, T_N
        self.alpha = alpha
        self.n_action = S_N.n_action

        self.learn_step_counter = 1
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.device = device
        self.eval_net = self.eval_net.to(device)
        self.target_net = self.target_net.to(device)
        self.sch_lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 1, gamma = 0.1 ** 0.1)

    def get_value(self, state):

        _, Q_l = self.act(state, debug = False)
        return Q_l

    def act(self, state, debug = False):
        action_l = []
        Q_l = []
        Q = self.eval_net(state)
        Q2 = self.target_net(state)
        for i in range(len(self.n_action)):
            ma_Q, _ = torch.max(Q[i], -1)
            Q[i] -= ma_Q.reshape(-1, 1)
            Q[i] /= self.alpha
            Q[i] = torch.exp(Q[i])
            Q[i] /= torch.sum(Q[i], -1).reshape(-1, 1)
            rnd = torch.randn((Q[i].shape[0], 1), device = self.device)
            Q[i] = Q[i] * (rnd < EPISILO) + torch.ones_like(Q[i]) * (rnd >= EPISILO)
            action = torch.multinomial(Q[i], 1)
            Q_l.append(Q2[i].gather(1, action))
            action_l.append(action)
        action_l = torch.cat(action_l, -1)
        Q_l = torch.cat(Q_l, -1)
        return action_l, Q_l
    
    def learn(self, rollout, n_minibatch, epoch, lr_decay = False):
        
        data_generator = rollout.feed_forward_generator(n_minibatch)
        if epoch >= 20 and epoch < 30:self.sch_lr.step()
        t_c = 0
        for sample in data_generator:
            loss = 0
            obs_b, action_b, return_b = sample
            obs_batch = obs_b.to(self.device)
            action_batch = action_b.to(self.device)
            return_batch = return_b.to(self.device)
            q_eval = self.eval_net(obs_batch)
            for i in range(len(self.n_action)):
                q_eval_i = q_eval[i].gather(1, action_batch[:,i:i+1]).squeeze()
                loss = loss + 0.5 * (return_batch[:,i].detach() - q_eval_i).pow(2).mean() / len(self.n_action)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            if t_c % 10 == 0:
                print('training loss', loss)
            t_c += 1
        
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1
        global EPISILO 
        EPISILO += 0.1
        EPISILO = min(EPISILO, 1.2)