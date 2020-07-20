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
    def __init__(self, module, module_copy, env, lr=1e-2, episodes=100000, copy_every=100, gamma=0.89):
        self.module = module.to(device)
        self.module.gamma = gamma
        self.module.load_state_dict(torch.load("./params_self_training.pth"))
        self.partner = module_copy.to(device)
        self.partner.gamma = gamma
        self.partner.load_state_dict(self.module.state_dict())
        self.optim = optim.Adam(self.module.parameters(), lr=lr)
        self.episodes = episodes
        self.env = env
        self.rows = env.configuration['rows']
        self.columns = env.configuration['columns']
        self.copy_every = copy_every
        print(self.module.gamma)
        print(self.partner.gamma)
    
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
            if "board" in observation:
                observation = observation["board"]
            action, _ = self.partner.get_action(observation)
            #print("Partner's move: {}".format(action))
            return action
        
        all_rewards = []
        
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

            #print("Player_pos: {}".format(player_pos))
                
            observation = trainer.reset()
            t = 0
            while not self.env.done:
                #print("Round: {}".format(t))
                switch_back = False
                if observation.mark != 1:
                    #print("Switch marks")
                    switch_back = True
                    observation['board'] = switch(observation['board'])
                    
                action, log_prob = self.module.get_action(observation['board'])
                #print("My move ({}): {}".format("blue" if observation.mark == 1 else "grey", action))

                if switch_back:
                    #print("Switch back marks")
                    observation['board'] = switch(observation['board'])
                
                observation, reward, done, _ = trainer.step(action)

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
                t += 1
                    
            self.module.update_policy(rewards, log_probs)
            all_rewards.append(np.sum(np.array(rewards)))
            #print(rewards)
            
            if episode % self.copy_every == 0:
                print("Update parameters of partner module")
                self.partner.load_state_dict(self.module.state_dict())
                last_200_rewards = np.array(all_rewards[max(0, episode-200) : (episode+1)])
                plt.plot(last_200_rewards)
                plt.xlabel('Episode')
                plt.savefig("./rewards_last_200_episodes.pdf")
                plt.close()
                torch.save(self.module.state_dict(), "./params_self_training.pth")
        plt.figure()
        plt.plot(np.cumsum(all_rewards), label="Cumulated Reward")
        plt.xlabel('Episode')
        plt.title('Self Training:\nUpdate of partner agent every {} episodes'.format(self.copy_every))
        plt.legend()
        plt.savefig("./Cumulated_rewards_complete_training.pdf")
        plt.close()
        torch.save(self.module.state_dict(), "./params_self_training.pth")
        


if __name__ == "__main__":
    env = make("connectx", debug=False)
    cols = env.configuration.columns
    rows = env.configuration.rows
    NetworkTrainer(PolicyNetwork(cols*rows, cols), PolicyNetwork(cols*rows, cols), env).training()

