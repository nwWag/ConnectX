import torch
from torch import nn, optim
import numpy as np
from config import device
import torch.nn.functional as F
from torch.autograd import Variable

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++      
# File to store neural network agents. We tried various architectures. The best have been Q_Network (CNN,SELU and wihtout batchnorm) 
# and ...
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Q_Network_Conv(nn.Module):
    def __init__(self, nc=1, ngf = 16):
        super(Q_Network_Conv, self).__init__()
        self.ngf = ngf
        self.conv = nn.Sequential(
            nn.Conv2d( nc, ngf, 5, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.SELU(True),

            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.SELU(True),

            nn.Conv2d( ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.SELU(True)
        )

        self.predict = nn.Sequential(nn.Linear(ngf + 4, 1), nn.Tanh())




    def forward(self, x, action, connect_x=torch.ones(1).to(device) * 4):

        config = torch.unsqueeze(torch.cat((torch.ones(1).to(device) * x.shape[2], torch.ones(1).to(device) * x.shape[3], action, connect_x), dim=0), dim=0)
        config = config.repeat(x.shape[0], 1)
        x = self.conv(x.float())
        x = x.reshape(x.shape[0], self.ngf, -1)
        x, _ = torch.max(x, dim=2)
        x = self.predict(torch.cat((x, config), dim=1))

        return x

class Q_Network_Flat(nn.Module):
    def __init__(self, nc=6*7, ngf = 128, actions=7):
        super(Q_Network_Flat, self).__init__()
        self.ngf = ngf
        self.predict = nn.Sequential(nn.Linear(nc, ngf), nn.SELU(),
                                    nn.Linear(ngf, ngf), nn.SELU(),
                                    nn.Linear(ngf, actions))#, nn.Tanh())




    def forward(self, x):
        x = x.reshape(x.shape[0], -1).float()
        x = self.predict(x)

        return x

class Q_Network(nn.Module):
    def __init__(self, ngf = 64, actions=7, bn=False):
        super(Q_Network, self).__init__()
        self.ngf = ngf
        self.activation = nn.PReLU
        self.bn = bn

        self.conv = nn.Sequential(
            nn.Conv2d(3, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(3) if self.bn else nn.Identity(),
            self.activation(),

            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(3) if self.bn else nn.Identity(),
            self.activation()
        )

        self.predict = nn.Sequential(nn.Linear(126, ngf), nn.BatchNorm1d(ngf) if self.bn else nn.Identity(), self.activation(),
                                     nn.Linear(ngf, ngf), nn.BatchNorm1d(ngf) if self.bn else nn.Identity(), self.activation(),
                                     nn.Linear(ngf, actions))#, nn.Softmax())

    def forward(self, x):
        rows = x.shape[2]
        columns = x.shape[3]
        x_new = torch.empty(x.shape[0], 3, rows, columns).to(device)
        for i in range(x.shape[0]):
            x_new[i] = F.one_hot(x[i].reshape(rows, columns).long(), 3).permute(2,0,1)

        x = x_new
        x = x.reshape(x.shape[0], -1).float()
        x = self.predict(x)

        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=128, learning_rate=3e-4, gamma=0.90):
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
        probs = self.forward(Variable(state))  # probs have shape (1 x num_actions)
        red_choices = torch.IntTensor([c for c in range(self.num_actions) if state.squeeze(0).numpy()[c] == 0])
        red_probs = probs.detach()[0, row == 0] / torch.sum(probs.detach()[0, row == 0])
        highest_prob_action = int(np.random.choice(red_choices, p=red_probs.numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        if discounted_rewards.size(0) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                        discounted_rewards.std() + 1e-9)  # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()
