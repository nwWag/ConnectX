#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from kaggle_environments import evaluate, make, utils
from tqdm import tqdm
from agents import PolicyNetwork
import torch


# training:
def train():
    env = make("connectx", debug=False)
    trainer = env.train([None, "negamax"])
    
    net1 = PolicyNetwork(env.configuration.columns*env.configuration.rows, env.configuration.columns, 128)
    net1.load_state_dict(torch.load("model/REINFORCE_params.pth"))
        
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
    #torch.save(net1.state_dict(), "model/REINFORCE_params_old.pth")
        
    plt.plot(np.cumsum(all_rewards), label="Cumulated Reward")
    plt.xlabel('Episode')
    plt.legend()
    plt.show()


def main():
    train()


if __name__ == '__main__':
    main()

