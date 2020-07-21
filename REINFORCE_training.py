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
from REINFORCE_Agent import PolicyNetwork


# training:
def train():
    env = make("connectx", configuration= {'rows': 6, 'columns': 8, 'inarow': 4}, debug=True)
    trainer = env.train([None, "random"])
    
    net1 = PolicyNetwork(env.configuration.columns*env.configuration.rows, env.configuration.columns, 128)
    #net1.load_state_dict(torch.load("./params.pth"))
        
    max_episode_num = 5000
    max_steps = 10000
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in tqdm(range(max_episode_num)):
        state = trainer.reset().board
        
        log_probs = []
        rewards = []

        for steps in range(max_steps):
            action, log_prob = net1.get_action(state)
            new_state, reward, done, _ = trainer.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                net1.update_policy(rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                break
            
            state = new_state.board
    
    # save model parameters
    torch.save(net1.state_dict(), "./params.pth")
        
    plt.plot(np.cumsum(all_rewards), label="Cumulated Reward")
    plt.xlabel('Episode')
    plt.legend()
    plt.show()


def main():
    train()


if __name__ == '__main__':
    main()

