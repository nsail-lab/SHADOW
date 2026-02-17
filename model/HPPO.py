import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from collections import deque


class PPOMemory:
    def __init__(self, max_size, history_len, continuous_action_dim, cuda="0"):
        self.max_size = max_size
        self.history_len = history_len
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.buffer = []
        self.continuous_action_dim = continuous_action_dim

    def store_memory(self, state, discrete_action, continuous_action, log_probs, value, reward, done):
        self.buffer.append((state, discrete_action, continuous_action, log_probs, value, reward, done))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        buffer_len = len(self.buffer)
        indices = np.random.choice(buffer_len, batch_size, replace=buffer_len < batch_size)

        states_batch, d_actions_batch, c_actions_batch = [], [], []
        log_probs_d_batch, log_probs_c_batch = [], []
        vals_batch, rewards_batch, dones_batch = [], [], []

        for idx in indices:
            start_idx = max(0, idx - self.history_len + 1)
            state_seq = [self.buffer[i][0] for i in range(start_idx, idx + 1)]
            state_seq = [state_seq[0]] * (self.history_len - len(state_seq)) + state_seq
            states_batch.append(state_seq)
            d_actions_batch.append(self.buffer[idx][1])
            c_actions_batch.append(self.buffer[idx][2])
            log_probs_d_batch.append(self.buffer[idx][3]['discrete'])
            log_probs_c_batch.append(self.buffer[idx][3]['continuous'])
            vals_batch.append(self.buffer[idx][4])
            rewards_batch.append(self.buffer[idx][5])
            dones_batch.append(self.buffer[idx][6])

        return (
            torch.tensor(states_batch, dtype=torch.float32).to(self.device),
            torch.tensor(d_actions_batch, dtype=torch.int64).to(self.device),
            torch.tensor(c_actions_batch, dtype=torch.float32).to(self.device),
            torch.tensor(log_probs_d_batch, dtype=torch.float32).to(self.device),
            torch.tensor(log_probs_c_batch, dtype=torch.float32).to(self.device),
            torch.tensor(vals_batch, dtype=torch.float32).to(self.device),
            torch.tensor(rewards_batch, dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.tensor(dones_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        )


class HierarchicalActorNetwork(nn.Module):
    def __init__(self, input_dims, alpha, continuous_action_dim, cuda="0"):
        super().__init__()
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(*input_dims, 256, batch_first=True)
        self.discrete_head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2))
        self.continuous_head = nn.Sequential(nn.Linear(256 + 2, 256), nn.ReLU(), nn.Linear(256, continuous_action_dim))
        self.log_std = nn.Parameter(torch.zeros(continuous_action_dim))
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def forward(self, state_seq):
        _, (hn, _) = self.lstm(state_seq)
        base = hn[-1]
        logits = self.discrete_head(base)
        d_dist = Categorical(logits=logits)
        d_action = d_dist.sample()
        d_onehot = F.one_hot(d_action, num_classes=2).float()
        c_input = torch.cat([base, d_onehot], dim=-1)
        c_mean = self.continuous_head(c_input)
        c_std = self.log_std.exp().expand_as(c_mean)
        c_dist = Normal(c_mean, c_std)
        return d_dist, d_action, c_dist


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, cuda="0"):
        super().__init__()
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(*input_dims, 256, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def forward(self, state_seq):
        _, (hn, _) = self.lstm(state_seq)
        return self.fc(hn[-1])


class HPPOAgent:
    def __init__(self, input_dims, n_continuous_actions, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, history_len=10, batch_size=64, replay_buffer_size=1000, cuda="0"):
        self.gamma, self.policy_clip, self.gae_lambda = gamma, policy_clip, gae_lambda
        self.batch_size, self.history_len = batch_size, history_len
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

        self.actor = HierarchicalActorNetwork(input_dims, alpha, n_continuous_actions, cuda)
        self.critic = CriticNetwork(input_dims, alpha, cuda)
        self.memory = PPOMemory(replay_buffer_size, history_len, n_continuous_actions, cuda)
        self.state_history = deque(maxlen=history_len)

    def update_state_history(self, new_state):
        self.state_history.append(new_state)
        if len(self.state_history) < self.history_len:
            return np.array([self.state_history[0]] * (self.history_len - len(self.state_history)) + list(self.state_history))
        return np.array(self.state_history)

    def choose_action(self, state):
        state_seq = self.update_state_history(state)
        state_seq = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        d_dist, d_action, c_dist = self.actor(state_seq)
        c_action = c_dist.sample()

        log_probs = {
            'discrete': d_dist.log_prob(d_action).item(),
            'continuous': c_dist.log_prob(c_action).sum().item()
        }
        value = self.critic(state_seq).item()

        action = {
            'discrete': d_action.item(),
            'continuous': c_action.squeeze().detach().cpu().numpy()
        }
        return action, log_probs, value

    def remember(self, state, action, log_probs, value, reward, done):
        self.memory.store_memory(state, action['discrete'], action['continuous'], log_probs, value, reward, done)

    def train(self):
        states, d_actions, c_actions, old_log_d, old_log_c, values, rewards, dones = self.memory.sample(self.batch_size)
        advantages = torch.zeros_like(rewards).to(self.device)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t+1] * mask if t < len(rewards) - 1 else rewards[t] - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * mask * last_adv
            last_adv = advantages[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        d_dist, _, c_dist = self.actor(states)
        new_log_d = d_dist.log_prob(d_actions)
        new_log_c = c_dist.log_prob(c_actions).sum(dim=-1)

        ratio_d = torch.exp(new_log_d - old_log_d)
        ratio_c = torch.exp(new_log_c - old_log_c)

        surr1_d = ratio_d * advantages.squeeze()
        surr2_d = torch.clamp(ratio_d, 1 - self.policy_clip, 1 + self.policy_clip) * advantages.squeeze()
        loss_actor_d = -torch.min(surr1_d, surr2_d).mean()

        surr1_c = ratio_c * advantages.squeeze()
        surr2_c = torch.clamp(ratio_c, 1 - self.policy_clip, 1 + self.policy_clip) * advantages.squeeze()
        loss_actor_c = -torch.min(surr1_c, surr2_c).mean()

        critic_values = self.critic(states).squeeze()
        loss_critic = F.mse_loss(returns.squeeze(), critic_values)

        total_loss = loss_actor_d + loss_actor_c + 0.5 * loss_critic
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
