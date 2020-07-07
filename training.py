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

class NetworkTrainer():
    def __init__(self, module, module_copy, env, lr=1e-2, episodes=100000, epochs=2, eps=.99, loss=nn.SmoothL1Loss(), 
                gamma=0.99, batch_size=64, replay_size= 15000, eval_every=3000, copy_every=150):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++      
        # Settings +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.module = module.to(device) # Policy Network    
        self.module_hat = module_copy.to(device) # Target Network
        self.module_hat.load_state_dict(self.module.state_dict()) # Init both the same
        self.optim = optim.Adam(self.module.parameters(), lr=lr) # Use Adam. Its the best.
        self.episodes = episodes # Number of games to paly
        self.epochs = epochs # Number of epochs to train
        self.env = env # Enviroment to play
        self.loss = loss # Loss to measure difference of Q values
        self.rows = env.configuration['rows'] # Rows of playing field
        self.columns = env.configuration['columns'] # Columns of playing field
        self.eps = eps # epsilo for epsilon greedy search
        self.replay_memory = [] # Store seen transitions
        self.gamma = gamma # Weight current vs other rewards
        self.batch_size = batch_size 
        self.replay_size = replay_size # How many transitions to store
        self.best = -50 # TODO: Delete
        self.eval_every = eval_every # How often to evaluate. Not implemented anymore, seperated testing.
        self.copy_every = copy_every # How often to train and update target network


    def training(self):
        self.env.reset()
        self.Qs = None
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++      
        # Utils functions ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # Get Q values for full batch and decide wether to use policy or target network
        def get_Qs_batch(input_observation, hat=False):       
            if hat:
                Qs = self.module_hat(input_observation)
            else:  
                Qs = self.module(input_observation)
            self.Qs = Qs
            return Qs

        # Get Q values for one transition
        def get_Qs(observation):
            input_observation = torch.from_numpy(np.array(observation['board'])).to(device).reshape(1,1, self.rows, self.columns)                
            Qs = self.module(input_observation)
            self.Qs = Qs[0]
            return Qs[0]

        # Change marks in case our agent starts at the second position
        def switch(board):
            board[board == 1] = 3
            board[board == 2] = 1
            board[board == 3] = 2
            return board
        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++      
        # Agent functions ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # Temporary agent that does not use lookahead search but only q values
        def temp_agent(observation, _):
            return int(np.argmax(np.array([(q.item() if observation['board'][c] == 0 else -float('inf')) for c, q in enumerate(get_Qs(observation))])))

        # Store rewards to observe training progress
        cum_rewards = []
        avg_rewards = []

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++      
        # Run episodes +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        for episode in range(self.episodes):

            cum_reward = 0
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

                # Decide step ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                self.module.eval()
                if observation.mark != 1:
                    observation['board'] = switch(observation['board'])
                observation_old = observation
                if random.uniform(0, 1) <= self.eps:
                    action = choice([c for c in range(self.columns) if observation['board'][c] == 0])
                else:
                    with torch.no_grad():
                        action = temp_agent(observation, None)
                self.module.train()

                # Update eps  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                self.eps = self.eps * 0.99999
                
                # Conduct step +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                t += 1
                observation, reward, done, info = trainer.step(action) 

                # Custom rewards to penalize long runs +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                reward = 0 if reward is None else reward # Avoid illegal things.
                if done:
                    if reward == 1: 
                        reward = 20
                    else: 
                        reward = 0
                else:
                    reward = -0.05
                cum_reward += reward

                # Store transistions ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                self.replay_memory.append((np.array(observation_old['board']).reshape(1, self.rows, self.columns), action, reward,
                                            np.array(observation['board']).reshape(1, self.rows, self.columns), done, player_pos))
                # Check replay size +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if len(self.replay_memory) > self.replay_size:
                    del self.replay_memory[0]

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Store network params and plot +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            cum_rewards.append(np.array(cum_reward))
            avg_rewards.append(np.mean(np.array(cum_rewards[max(0, episode - 200):(episode + 1)])))
            if avg_rewards[-1] > self.best and episode > 200:
                self.env.reset()
            torch.save(self.module.state_dict(), "model/"+ type(self.module).__name__ +  ".pt")
            torch.save(self.optim.state_dict(), "optimizer/"+ type(self.module).__name__  + ".pt")
            self.best = avg_rewards[-1]

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Conduct actual gradient descent +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if episode % self.copy_every == self.copy_every - 1:
                # Plot rewards ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Do not plot to often due to performance
                plt.plot(avg_rewards)
                plt.savefig('plots/cum_rewards.png')
                plt.close()

                loss_arr = []
                for epoch in range(self.epochs):
                    # Select transitions and concatenate to batch +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    for batch_pointer in range(0, len(self.replay_memory), self.batch_size):
                        batch_tuples = self.replay_memory[batch_pointer:(batch_pointer + self.batch_size)]
                        if len(batch_tuples) < self.batch_size:
                            continue

                        observation_batch = torch.stack([torch.from_numpy(np.array(tuple_[3])).to(device).float() for tuple_ in batch_tuples])
                        observation_old_batch = torch.stack([torch.from_numpy(np.array(tuple_[0])).to(device).float() for tuple_ in batch_tuples])
                        reward_batch = torch.stack([torch.from_numpy(np.array(tuple_[2])).to(device).float() for tuple_ in batch_tuples])

                        action_batch = torch.from_numpy(np.array([[i, tuple_[1]] for i, tuple_ in enumerate(batch_tuples)]))
                        done_batch = np.array([tuple_[4] for tuple_ in batch_tuples])


                        # Calculate target ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        # Using Double DQN implementation
                        # Make use of zero sum condition and use negative Q values of opponent as target
                        # (see https://www.kaggle.com/matant/pytorch-dqn-connectx for a discussion of this)
                        with torch.no_grad():
                            Qs_batch_hat = -get_Qs_batch(switch(observation_batch), hat=True)
                            Qs_batch = get_Qs_batch(observation_batch, hat=False)

                            # Check if column full ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            illegal = torch.squeeze(observation_batch, dim=1)[torch.arange(Qs_batch.shape[0]).to(device),0] != 0
                            # Be good, dont do anything illegal
                            Qs_batch[illegal] = -1000
                            Qs_batch_hat[illegal] = -1000
                            # Check where "to go" to update Q value +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            Qs_arg_max_batch = torch.argmax(Qs_batch, dim=1)

                            # Calculate target with current reward ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

if __name__ == "__main__":
    env = make("connectx", debug=False)
    env.render()
    NetworkTrainer(Q_Network(), Q_Network(), env).training()