import torch
from torch import nn, optim
import numpy as np
from config import device
import random
from random import choice
from kaggle_environments import evaluate, make, utils
from agents import Q_Network

def mean_reward(rewards):
    return sum(r[0] if r[0] is not None else -1 for r in rewards) / float(len(rewards))


class NetworkTrainer():
    def __init__(self, module, env, lr=1e-3, episodes=500, eps=0.5, loss=nn.MSELoss()):
        self.module = module.to(device)
        self.optim = optim.Adam(self.module.parameters(), lr=lr)
        self.episodes = episodes
        self.env = env
        self.rows = env.configuration['rows']
        self.columns = env.configuration['columns']
        self.eps = eps
        self.loss_f = loss



    def training(self, print_eval=True):
        self.env.reset()
        trainer = self.env.train([None, "negamax"])
        self.Qs = None
        def get_Qs(observation):
            Qs  = []
            input_observation = torch.from_numpy(np.array(observation['board'])).to(device).reshape(1,1, self.rows, self.columns)                
            for c in range(self.columns):
                Q_c = self.module(input_observation, torch.ones(1).to(device) * c)
                Qs.append(Q_c)
            self.Qs = Qs
            return Qs

        def temp_agent(observation, _):
            return int(np.argmax(np.array([q.item() for q in get_Qs(observation)])))

        for episode in range(self.episodes):
        
            observation = trainer.reset()
            t = 0
            while not self.env.done:
                observation_old = observation
                if random.uniform(0, 1) <= self.eps:
                    action = choice([c for c in range(self.columns) if observation.board[c] == 0])
                else:
                    with torch.no_grad():
                        action = temp_agent(observation, None)
                    

                #print("Episode", episode, "t", t,"Action", action)
                t += 1
                observation, reward, done, info = trainer.step(action)
                reward = -1 if reward is None else reward

                value_hypo = 0
                if not done:
                    with torch.no_grad():
                        hypothetical_action = temp_agent(observation, None)
                        value_hypo = self.Qs[hypothetical_action]

                y = torch.ones(1).to(device) * (reward + value_hypo)

                action_old = temp_agent(observation_old, None)
                x = self.Qs[action_old]
                self.optim.zero_grad()
                loss = self.loss_f(x, y)
                loss.backward()
                self.optim.step()

            
            if print_eval:
                self.env.reset()
                print(loss)
                print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [temp_agent, "negamax"], num_episodes=10)))
                print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [temp_agent, "random"], num_episodes=10)))
    

if __name__ == "__main__":
    env = make("connectx", debug=True)
    env.render()
    NetworkTrainer(Q_Network(), env).training()