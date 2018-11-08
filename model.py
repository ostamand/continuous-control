import torch
import torch.nn as nn 
import torch.nn.functional as F

class GaussianActorCritic(nn.Module):

    def __init__(self, state_size, action_dim):
        super(GaussianActorCritic, self).__init__()
        self.state_size = state_size
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_size, 100)
        self.fc2 = nn.Linear(100, 100)

        self.fc_actor = nn.Linear(100, self.action_dim)
        self.fc_critic = nn.Linear(100, 1)

        self.std = nn.Parameter(torch.zeros(1, action_dim))
    
    def forward(self, x, action=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Actor
        mean = torch.tanh(self.fc_actor(x))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        # Critic
        # State value V(s)
        v = self.fc_critic(x)

        return action, log_prob, dist.entropy(), v

class CrawlerActorCritic(nn.Module):

    def __init__(self, state_size, action_dim):
        super(CrawlerActorCritic, self).__init__()
        self.state_size = state_size
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_size, 100)
        self.fc2 = nn.Linear(100, 100)

        self.fc_actor = nn.Linear(100, self.action_dim)
        self.fc_critic = nn.Linear(100, 1)

        self.std = nn.Parameter(torch.zeros(1, action_dim))
    
    def forward(self, x, action=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Actor
        mean = torch.tanh(self.fc_actor(x))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        # Critic
        # State value V(s)
        v = self.fc_critic(x)

        return action, log_prob, dist.entropy(), v
