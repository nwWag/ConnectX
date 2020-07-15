#!/usr/bin/env python
# coding: utf-8

import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from kaggle_environments import evaluate, make, utils
from tqdm import tqdm
import copy


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=128, learning_rate=3e-4, gamma=0.99):
        super(PolicyNetwork, self).__init__()
        
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = gamma
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x
    
    def get_action(self, state):
        row = torch.tensor([state[c] for c in range(self.num_actions)])
        state = torch.from_numpy(np.asarray(state)).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        red_choices = torch.IntTensor([c for c in range(self.num_actions) if state.squeeze(0).numpy()[c] == 0])
        red_probs = probs.detach()[0, row == 0] / torch.sum(probs.detach()[0, row == 0])
        highest_prob_action = int(np.random.choice(red_choices, p=red_probs.numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob
    
    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        #log_probs = torch.tensor(log_probs)
    
        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
    
        discounted_rewards = torch.tensor(discounted_rewards)
        if discounted_rewards.size(0) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
            
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
    
        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step() 

