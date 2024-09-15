import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, ob_dim, ac_dim,hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(ob_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, ac_dim)
        self.fc_std = nn.Linear(hidden_dim, ac_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x)) * 2
        std = self.softplus(self.fc_std(x)) + 1e-5
        return mean, std

    def sample_action(self, x):
        mean, std = self.forward(x)
        dist = D.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -2, 2)
        return action

class Critic(nn.Module):
    def __init__(self, ob_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(ob_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class ClipPPO:
    def __init__(self, ob_dim, ac_dim, gamma=0.99, Lambda = 0.95 ,clip_ratio=0.2, lr_actor=3e-4, lr_critic=1e-3):
        self.actor = Actor(ob_dim, ac_dim).to(device)
        self.old_actor = Actor(ob_dim, ac_dim).to(device)
        self.critic = Critic(ob_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.Lambda = Lambda
        self.clip_ratio = clip_ratio
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.epochs = 10
        self.memory = []
        self.advantages = []

    def store(self,observation, action, reward, value, done):
        reward = (reward + 8.0) / 8.0
        self.memory.append((observation, action, reward, value, done))

    def clear_memory(self):
        self.memory = []
        self.advantages = []

    def get_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32).to(device)
        action = self.actor.sample_action(observation)
        value = self.critic(observation)
        return action.cpu().detach().numpy(), value.cpu().detach().numpy()


    def update(self):
        self.old_actor.load_state_dict(self.actor.state_dict())
        obs, actions, rewards, values, dones = zip(*self.memory)
        T = len(rewards)
        for t in range(T-1):
            delta_t = rewards[t] + self.gamma * (1 - dones[t]) * values[t + 1] - values[t]
            self.advantages.append(delta_t)

        advantage_tensor = torch.tensor(self.advantages, dtype=torch.float32).to(device)
        value_tensor = torch.tensor(values, dtype=torch.float32).to(device)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        action_tensor = torch.tensor(actions, dtype=torch.float32).to(device)

        for t in range(T-1):
            with torch.no_grad():
                if(t==0):
                    old_mu, old_sigma = self.old_actor(obs_tensor[t])
                    old_pi = D.Normal(old_mu, old_sigma)
                    log_probs_old = old_pi.log_prob(action_tensor[t])

            mu, sigma = self.actor(obs_tensor[t])
            pi = D.Normal(mu, sigma)
            log_probs = pi.log_prob(action_tensor[t])

            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantage_tensor[t]
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage_tensor[t]

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(value_tensor[t]+advantage_tensor[t], self.critic(obs_tensor[t]))

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        self.clear_memory()

    def save_policy(self, path):
        torch.save(self.actor.state_dict(), path)
