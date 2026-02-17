import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque


class PPOMemory:
    def __init__(self, max_size, history_len, cuda="0"):
        self.max_size = max_size
        self.buffer = []
        self.history_len = history_len
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    def sample(self, batch_size):
        buffer_len = len(self.buffer)
        if buffer_len < batch_size:
            batch_idx = np.random.choice(np.arange(buffer_len), batch_size, replace=True)
        else:
            batch_idx = np.random.choice(buffer_len, batch_size, replace=False)

        states_batch, actions_batch, probs_batch, vals_batch, rewards_batch, dones_batch = [], [], [], [], [], []
        for idx in batch_idx:
            start_idx = max(0, idx - self.history_len + 1)
            state_sequence = [self.buffer[i][0] for i in range(start_idx, idx + 1)]
            state_sequence = [state_sequence[0]] * (self.history_len - len(state_sequence)) + state_sequence

            states_batch.append(state_sequence)
            actions_batch.append(self.buffer[idx][1])
            probs_batch.append(self.buffer[idx][2])
            vals_batch.append(self.buffer[idx][3])
            rewards_batch.append(self.buffer[idx][4])
            dones_batch.append(self.buffer[idx][5])

        # Convert to tensors
        states_batch = torch.FloatTensor(np.array(states_batch)).to(self.device)
        actions_batch = torch.FloatTensor(np.array(actions_batch)).to(self.device)
        probs_batch = torch.FloatTensor(np.array(probs_batch)).to(self.device)
        vals_batch = torch.FloatTensor(np.array(vals_batch)).to(self.device)
        rewards_batch = torch.FloatTensor(np.array(rewards_batch).reshape(batch_size, -1)).to(self.device)
        dones_batch = torch.FloatTensor(np.array(dones_batch).reshape(batch_size, -1)).to(self.device)

        return states_batch, actions_batch, probs_batch, vals_batch, rewards_batch, dones_batch

    def store_memory(self, state, action, probs, vals, reward, done):
        experience = (state, action, probs, vals, reward, done)
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def __len__(self):
        return len(self.buffer)


class ContinuousActorNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, action_dim=1, chkpt_dir='tmp/ppo', cuda="0"):
        super(ContinuousActorNetwork, self).__init__()
        self.lstm = nn.LSTM(*input_dims, fc1_dims, batch_first=True)
        self.fc1 = nn.Linear(fc1_dims, fc1_dims)
        self.mu = nn.Linear(fc1_dims, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # learnable log std dev

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state_sequence):
        lstm_out, (hn, cn) = self.lstm(state_sequence)
        x = torch.relu(self.fc1(hn[-1]))
        mu = self.mu(x)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        return dist

    def save_checkpoint(self, filename):
        torch.save(self.state_dict(), filename)

    def load_checkpoint(self, filename):
        self.load_state_dict(torch.load(filename, map_location=self.device))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo', cuda="0"):
        super(CriticNetwork, self).__init__()
        self.lstm = nn.LSTM(*input_dims, fc1_dims, batch_first=True)
        self.fc1 = nn.Linear(fc1_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state_sequence):
        lstm_out, (hn, cn) = self.lstm(state_sequence)
        x = torch.relu(self.fc1(hn[-1]))
        value = self.fc2(x)
        return value
    
    def save_checkpoint(self, filename):
        torch.save(self.state_dict(), filename)

    def load_checkpoint(self, filename):
        self.load_state_dict(torch.load(filename, map_location=self.device))


class PPOContinuousAgent:
    def __init__(self, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, history_len=10, batch_size=64,
                 replay_buffer_size=1000, cuda="0"):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.history_len = history_len

        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

        self.actor = ContinuousActorNetwork(input_dims, alpha, cuda=cuda)
        self.critic = CriticNetwork(input_dims, alpha, cuda=cuda)
        self.memory = PPOMemory(replay_buffer_size, history_len=history_len, cuda=cuda)
        self.state_history = deque(maxlen=self.history_len)

    def clear_history(self):
        self.state_history.clear()

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save(self, save_dir):
        self.actor.save_checkpoint(os.path.join(save_dir, "PPO_actor_evader"))
        self.critic.save_checkpoint(os.path.join(save_dir, "PPO_critic_evader"))

    def load(self, load_dir):
        self.actor.load_checkpoint(os.path.join(load_dir, "PPO_actor_evader"))
        self.critic.load_checkpoint(os.path.join(load_dir, "PPO_critic_evader"))

    def update_state_history(self, new_state):
        self.state_history.append(new_state)
        if len(self.state_history) < self.history_len:
            first_state = self.state_history[0]
            padding_states = [first_state] * (self.history_len - len(self.state_history))
            return np.array(padding_states + list(self.state_history))
        else:
            return np.array(list(self.state_history))

    def choose_action(self, state):
        state_sequence = self.update_state_history(state)
        state_sequence = torch.tensor(state_sequence, dtype=torch.float32).to(self.device)

        dist = self.actor(state_sequence)
        value = self.critic(state_sequence)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return (
            action.squeeze().item(),
            log_prob.squeeze().item(),
            value.squeeze().item()
        )

    def train(self):
        states_batch, actions_batch, probs_batch, vals_batch, rewards_batch, dones_batch = \
            self.memory.sample(batch_size=self.batch_size)

        # normalize rewards
        rewards_batch = (rewards_batch - rewards_batch.mean()) / (rewards_batch.std() + 1e-8)

        advantages = torch.zeros(len(rewards_batch), dtype=torch.float32, device=self.device)
        last_advantage = 0
        for t in reversed(range(len(rewards_batch))):
            if t == len(rewards_batch) - 1:
                next_value = 0
                next_non_terminal = 0
            else:
                next_value = vals_batch[t + 1]
                next_non_terminal = 1 - dones_batch[t]
            delta = rewards_batch[t] + self.gamma * next_value * next_non_terminal - vals_batch[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            last_advantage = advantages[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dist = self.actor(states_batch)
        critic_value = self.critic(states_batch).squeeze()

        new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
        prob_ratio = torch.exp(new_log_probs - probs_batch)

        weighted_probs = advantages * prob_ratio
        weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
        actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

        returns = advantages + vals_batch
        critic_loss = (returns - critic_value) ** 2
        critic_loss = critic_loss.mean()

        total_loss = actor_loss + 0.5 * critic_loss
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
