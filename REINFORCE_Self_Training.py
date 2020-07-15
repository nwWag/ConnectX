#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn, optim
import numpy as np
from config import device
import random
from random import choice
from kaggle_environments import evaluate, make, utils
from REINFORCE_Agent import PolicyNetwork
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import copy
from tqdm import tqdm


class NetworkTrainer():
    def __init__(self, module, module_copy, env, lr=1e-2, episodes=2, gamma=0.99, 
                eval_every = 3000, copy_every=100):
        self.module = module.to(device)
        self.partner = module_copy.to(device)
        self.partner.load_state_dict(self.module.state_dict())
        self.optim = optim.Adam(self.module.parameters(), lr=lr)
        self.episodes = episodes
        self.env = env
        self.rows = env.configuration['rows']
        self.columns = env.configuration['columns']
        self.gamma = gamma
        self.eval_every = eval_every
        self.copy_every = copy_every
    
    def training(self):
        self.env.reset()
        
        # Change marks in case our agent is second player
        def switch(board):
            board_is_list = type(board) is list
            if board_is_list:
                board = np.array(board)
            board[board == 1] = 3
            board[board == 2] = 1
            board[board == 3] = 2
            if board_is_list:
                board = list(board)
            return board
            
        # function returning action of partner module  
        def temp_agent(observation, _):
            action, _ = self.partner.get_action(observation)
            return action
        
        cum_rewards = []
        
        for episode in tqdm(range(self.episodes)):
            log_probs = []
            rewards = []
            
            # Decide "who" starts to be able to start in first and second place
            if episode % 2 == 0:
                trainer = self.env.train([None, temp_agent])
                player_pos = 1
            else:
                trainer = self.env.train([temp_agent, None])
                player_pos = 2
                
            observation = trainer.reset()
            t = 0
            
            while not self.env.done:
                modules_move = False
                switch_back = False
                if observation.mark != 1:
                    switch_back = True
                    observation['board'] = switch(observation['board'])
                    
                if observation.mark * player_pos in [1, 4]:
                    modules_move = True
                    action, log_prob = self.module.get_action(observation['board'])
                else: 
                    action = temp_agent(observation['board'], None)
                
                #if switch_back:
                 #   observation['board'] = switch(observation['board'])
                
                print(self.env.done)
                observation, reward, done, _ = trainer.step(action)
                print(self.env.done)
                
                if modules_move:
                    print("My move")
                    reward = 0 if reward is None else reward
                    if done:
                        if reward == 1:
                            reward = 20
                        else:
                            reward = 0
                    else:
                        reward = -0.05
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    
            self.module.update_policy(rewards, log_probs)
            cum_rewards.append(np.sum(np.array(rewards)))
            
            if episode % self.copy_every == 0:
                self.partner.load_state_dict(self.module.state_dict())
                last_200_cum_rewards = np.array(cum_rewards[max(0, episode-200) : (episode+1)])
                plt.plot(last_200_cum_rewards)
                plt.savefig("./cum_rewards.png")
                plt.close()
                torch.save(self.module.state_dict(), "./params2.pth")
                      
        torch.save(self.module.state_dict(), "./params2.pth")
        


if __name__ == "__main__":
    env = make("connectx", debug=False)
    cols = env.configuration.columns
    rows = env.configuration.rows
    NetworkTrainer(PolicyNetwork(cols*rows, cols), PolicyNetwork(cols*rows, cols), env).training()

