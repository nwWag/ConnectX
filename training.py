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
    def __init__(self, module, env, lr=1e-3, episodes=10000, eps=.5, loss=nn.MSELoss(reduction='mean'), 
                gamma=1.0, batch_size=16, replay_size= 3000):
        self.module = module.to(device)
        self.optim = optim.Adam(self.module.parameters(), lr=lr)
        self.episodes = episodes
        self.env = env
        self.rows = env.configuration['rows']
        self.columns = env.configuration['columns']
        self.eps = eps
        self.loss_f = loss
        self.replay_memory = []
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_size = 3000
        self.best = -50


    def training(self, print_eval=True):
        self.env.reset()
        self.Qs = None
        # Agent functions ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def get_Qs_iter(observation):
            Qs  = []
            input_observation = torch.from_numpy(np.array(observation['board'])).to(device).reshape(1,1, self.rows, self.columns)                
            for c in range(self.columns):
                Q_c = self.module(input_observation, torch.ones(1).to(device) * c)
                Qs.append(Q_c)
            self.Qs = Qs
            return Qs

        def get_Qs_batch(input_observation):         
            Qs = self.module(input_observation)
            self.Qs = Qs
            return Qs


        def get_Qs(observation):
            input_observation = torch.from_numpy(np.array(observation['board'])).to(device).reshape(1,1, self.rows, self.columns)                
            Qs = self.module(input_observation)
            self.Qs = Qs[0]
            return Qs[0]

        def temp_agent(observation, _):
            return int(np.argmax(np.array([q.item() for q in get_Qs(observation)])))

        # Run episodes +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        trainer = self.env.train([None, temp_agent])
        for episode in range(self.episodes):
        
            observation = trainer.reset()
            t = 0
            while not self.env.done:
            
                # Decide step ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                observation_old = observation
                if random.uniform(0, 1) <= self.eps:
                    action = choice([c for c in range(self.columns) if observation.board[c] == 0])
                else:
                    with torch.no_grad():
                        action = temp_agent(observation, None)
                
                # Update eps
                self.eps = max(0.1, self.eps - (1.0/float(self.episodes)))

                #print("Episode", episode, "t", t,"Action", action)
                
                # Conduct step and store transition ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                t += 1
                observation, reward, done, info = trainer.step(action)
                reward = 0 if reward is None else reward
                reward = -1 if reward is None else reward

                self.replay_memory.append((np.array(observation_old['board']).reshape(1, self.rows, self.columns), action, reward,
                                            np.array(observation['board']).reshape(1, self.rows, self.columns), done))
                # Check replay size
                if len(self.replay_memory) > self.replay_size:
                    del self.replay_memory[0]

                # Select random transitions and concatenate to batch +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                batch_tuples = random.choices(self.replay_memory, k=self.batch_size)

                observation_batch = torch.stack([torch.from_numpy(np.array(tuple_[3])).to(device).float() for tuple_ in batch_tuples])
                observation_old_batch = torch.stack([torch.from_numpy(np.array(tuple_[0])).to(device).float() for tuple_ in batch_tuples])
                reward_batch = torch.stack([torch.from_numpy(np.array(tuple_[2])).to(device).float() for tuple_ in batch_tuples])

                action_batch = torch.from_numpy(np.array([[i, tuple_[1]] for i, tuple_ in enumerate(batch_tuples)]))
                done_batch = np.array([tuple_[4] for tuple_ in batch_tuples])

                # Calculate target +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                with torch.no_grad():
                    Qs_batch = get_Qs_batch(observation_batch)
                    Qs_max_batch, _ = torch.max(Qs_batch, dim=1)
                    Qs_max_batch[done_batch] = 0
                    y = reward_batch + self.gamma * Qs_max_batch

                # Perform gradient descent +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                Qs_old_batch =  get_Qs_batch(observation_old_batch)
                x = Qs_old_batch[action_batch[:,0], action_batch[:,1]]
                self.optim.zero_grad()
                loss = torch.mean((x - y)**2)
                loss.backward()
                self.optim.step()

            
            if print_eval and episode % 500 == 499:
                self.env.reset()
                print("Loss in episode", episode, loss.item())
                reward_random = mean_reward(evaluate("connectx", [temp_agent, "random"], num_episodes=50))
                reward_negamax =  mean_reward(evaluate("connectx", [temp_agent, "negamax"], num_episodes=50))
                print("Ours vs Negamax:", reward_negamax)
                print("Ours vs Random:",reward_random)
                print()

                if self.best < (reward_random+reward_negamax):
                    torch.save(self.module.state_dict(), "model/"+ type(self.module).__name__ +  ".pt")
                    torch.save(self.optim.state_dict(), "optimizer/"+ type(self.module).__name__  + ".pt")

if __name__ == "__main__":
    env = make("connectx", debug=False)
    env.render()
    NetworkTrainer(Q_Network(), env).training()