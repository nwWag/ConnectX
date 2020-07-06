import torch
from torch import nn, optim
import numpy as np
from config import device
import random
from random import choice
from kaggle_environments import evaluate, make, utils
from agents import Q_Network
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

def mean_reward(rewards):
    return sum(r[0] if r[0] is not None else -1 for r in rewards) / float(len(rewards))

class NetworkTrainer():
    def __init__(self, module, module_copy, env, lr=5e-4, episodes=100000, epochs=2, eps=.99, loss=nn.SmoothL1Loss(), 
                gamma=0.99, batch_size=32, replay_size= 15000, eval_every=2000, copy_every=100):
        self.module = module.to(device)
        self.module_hat = module_copy.to(device)
        self.module_hat.load_state_dict(self.module.state_dict())
        self.optim = optim.Adam(self.module.parameters(), lr=lr)
        self.episodes = episodes
        self.epochs = epochs
        self.env = env
        self.loss = loss
        self.rows = env.configuration['rows']
        self.columns = env.configuration['columns']
        self.eps = eps
        self.replay_memory = []
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.best = -50
        self.eval_every = eval_every
        self.copy_every = copy_every


    def training(self):
        self.env.reset()
        self.Qs = None
        # Agent functions ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def get_Qs_iter(observation):
            Qs  = [].N  
            input_observation = torch.from_numpy(np.array(observation['board'])).to(device).reshape(1,1, self.rows, self.columns)                
            for c in range(self.columns):
                Q_c = self.module(input_observation, torch.ones(1).to(device) * c)
                Qs.append(Q_c)
            self.Qs = Qs
            return Qs

        def get_Qs_batch(input_observation, hat=False):       
            if hat:
                Qs = self.module_hat(input_observation)
            else:  
                Qs = self.module(input_observation)
            self.Qs = Qs
            return Qs


        def get_Qs(observation):
            input_observation = torch.from_numpy(np.array(observation['board'])).to(device).reshape(1,1, self.rows, self.columns)                
            Qs = self.module(input_observation)
            self.Qs = Qs[0]
            return Qs[0]

        def temp_agent(observation, _):
            return int(np.argmax(np.array([(q.item() if observation['board'][c] == 0 else -float('inf')) for c, q in enumerate(get_Qs(observation))])))

        def switch(board):
            board[board == 1] = 3
            board[board == 2] = 1
            board[board == 3] = 2
            return board

        cum_rewards = []
        avg_rewards = []
        # Run episodes +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for episode in range(self.episodes):
            cum_reward = 0
            # Decide "who" starts to be able to start in first and second place
            if episode % 2 == 0:
                trainer = self.env.train([None, temp_agent])
                player_pos = 1
                switch_board = False
            else:
                trainer = self.env.train([temp_agent, None])
                player_pos = 2
                switch_board = True

            observation = trainer.reset()
            t = 0
            while not self.env.done:
                # Decide step ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                self.module.eval()
                observation_old = observation
                if switch_board:
                    observation_old['board'] = switch(observation['board'])
                    observation['board'] = switch(observation['board'])

                if random.uniform(0, 1) <= self.eps:
                    action = choice([c for c in range(self.columns) if observation['board'][c] == 0])
                else:
                    with torch.no_grad():
                        action = temp_agent(observation, None)
                self.module.train()

                # Update eps
                self.eps = self.eps * 0.99999

                #print("Episode", episode, "t", t,"Action", action)
                
                # Conduct step and store transition ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                t += 1
                observation, reward, done, info = trainer.step(action)
                reward = 0 if reward is None else reward
                if done:
                    if reward == 1: # Won
                        reward = 20
                    elif reward == 0: # Lost
                        reward = -20
                    else: # Draw
                        reward = 10
                else:
                    reward = -0.05
                    #reward = -1 if reward is None else reward

                cum_reward += reward
                self.replay_memory.append((np.array(observation_old['board']).reshape(1, self.rows, self.columns), action, reward,
                                            np.array(observation['board']).reshape(1, self.rows, self.columns), done, player_pos))
                # Check replay size
                if len(self.replay_memory) > self.replay_size:
                    del self.replay_memory[0]

            #Store anything to plot
            cum_rewards.append(np.array(cum_reward))
            avg_rewards.append(np.mean(np.array(cum_rewards[max(0, episode - 100):(episode + 1)])))

            # Update (use full memory) and copy 
            if episode % self.copy_every == self.copy_every - 1:
                plt.plot(avg_rewards)
                plt.savefig('plots/cum_rewards.png')
                plt.close()
                loss_arr = []
                # Select transitions and concatenate to batch +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                for epoch in range(self.epochs):
                    for batch_pointer in range(0, len(self.replay_memory), self.batch_size):
                        batch_tuples = self.replay_memory[batch_pointer:(batch_pointer + self.batch_size)]
                        if len(batch_tuples) < self.batch_size:
                            continue

                        observation_batch = torch.stack([torch.from_numpy(np.array(tuple_[3] if tuple_[5] == 1 else switch(tuple_[3]))).to(device).float() for tuple_ in batch_tuples])
                        observation_old_batch = torch.stack([torch.from_numpy(np.array(tuple_[0] if tuple_[5] == 1 else switch(tuple_[0]))).to(device).float() for tuple_ in batch_tuples])
                        reward_batch = torch.stack([torch.from_numpy(np.array(tuple_[2])).to(device).float() for tuple_ in batch_tuples])

                        action_batch = torch.from_numpy(np.array([[i, tuple_[1]] for i, tuple_ in enumerate(batch_tuples)]))
                        done_batch = np.array([tuple_[4] for tuple_ in batch_tuples])


                        # Calculate target +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        # Using Double DQN implementation
                        with torch.no_grad():
                            Qs_batch_hat = -get_Qs_batch(switch(observation_batch), hat=True)
                            Qs_batch = get_Qs_batch(observation_batch, hat=False)

                            # check if column full +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            illegal = torch.squeeze(observation_batch, dim=1)[torch.arange(Qs_batch.shape[0]).to(device),0] != 0
                            # Be good, dont do anything illegal
                            Qs_batch[illegal] = -1000
                            Qs_batch_hat[illegal] = -1000
                            Qs_arg_max_batch = torch.argmax(Qs_batch, dim=1)

                            Qs_max_batch = Qs_batch_hat[torch.arange(Qs_arg_max_batch.shape[0]).to(device),Qs_arg_max_batch]
                            Qs_max_batch[done_batch] = 0
                            y = reward_batch + self.gamma * Qs_max_batch


                        # Perform gradient descent +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        Qs_old_batch =  get_Qs_batch(observation_old_batch, hat=False)
                        x = Qs_old_batch[action_batch[:,0], action_batch[:,1]]
                        self.optim.zero_grad()
                        loss = self.loss(x, y)
                        loss.backward()
                        self.optim.step()
                        loss_arr.append(loss.item())
                
                print("After", episode, "episodes updated with", len(self.replay_memory), "transitions \nAverage loss", np.mean(np.array(loss_arr)), "\n", flush=True)
                # Copy +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                self.module_hat.load_state_dict(self.module.state_dict())


            # Eval and save in case the module has improved ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
            if episode % self.eval_every == self.eval_every -1:
                self.env.reset()
                reward_random = mean_reward(evaluate("connectx", [temp_agent, "random"], num_episodes=50))
                reward_negamax =  mean_reward(evaluate("connectx", [temp_agent, "negamax"], num_episodes=50))
                reward_rand_random = mean_reward(evaluate("connectx", ["random", "random"], num_episodes=10))
                reward_rand_negamax =  mean_reward(evaluate("connectx", ["random", "negamax"], num_episodes=10))
                print("Ours vs Negamax:", reward_negamax)
                print("Ours vs Random:",reward_random)
                print("Random vs Negamax:", reward_rand_negamax)
                print("Random vs Random:",reward_rand_random)
                print()

                if self.best < (reward_random+reward_negamax):
                    torch.save(self.module.state_dict(), "model/"+ type(self.module).__name__ +  ".pt")
                    torch.save(self.optim.state_dict(), "optimizer/"+ type(self.module).__name__  + ".pt")

if __name__ == "__main__":
    env = make("connectx", debug=False)
    env.render()
    NetworkTrainer(Q_Network(), Q_Network(), env).training()