
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.functional as F
import os

# Link to TD3 model
# https://github.com/sfujim/TD3/blob/master/TD3.py

class ReplayBuffer:
    def __init__(self, max_size, device):
        self.device = device
        self.max_size = max_size
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        buffer_len = len(self.buffer)
        if buffer_len < batch_size:
            # If buffer size is smaller than batch_size, sample with replacement
            batch_idx = np.random.choice(np.arange(buffer_len), batch_size, replace=True)
        else:
            batch_idx = np.random.choice(buffer_len, batch_size, replace=False)
            
        # Pad the batch with experiences from the same episode if necessary
        while len(batch_idx) < batch_size:
            batch_idx.extend(np.random.choice(buffer_len, batch_size - len(batch_idx), replace=False))
        batch = [self.buffer[idx] for idx in sorted(batch_idx)]
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states).reshape(batch_size, -1)).to(self.device)
        actions = torch.FloatTensor(np.array(actions).reshape(batch_size, -1)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(batch_size, -1)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states).reshape(batch_size, -1)).to(self.device)
        dones = torch.FloatTensor(np.array(dones).reshape(batch_size, -1)).to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.max_action * torch.tanh(self.layer3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
     
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    
class TD3Agent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 hidden_dim=256,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 expl_noise=0.1,
                 batch_size=256,
                 replay_buffer_size=100000,
                 cuda = "0"):
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.device)
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1.5e-6)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.expl_noise = expl_noise
        self.total_it = 0
        self.batch_size = batch_size

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def sample_action(self):
        return npr.uniform(low=-self.max_action, high=self.max_action, size=self.action_dim)

    def clear_history(self):
        pass
    
    def select_train_action(self, state):
        return (self.select_action(state) +
                npr.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                ).clip(-self.max_action, self.max_action)

    def train(self):
        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        #  print(f'DEBUG >>> state_sequence.shape = {state.shape}, next_state.shape = {next_state.shape}, action.shape = {action.shape}, reward.shape = {reward.shape}, done.shape = {done.shape}')
        
        reward = (reward-reward.mean())/(reward.std()+ 1e-8)
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1-done) * self.discount * target_Q
            # print(f'DEBUG >>> target_Q1.shape = {target_Q1.shape},target_Q2.shape = {target_Q2.shape}, target_Q.shape = {reward.shape}, target_Q1.shape = {reward.shape}')

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        # print(f'DEBUG >>> current_Q1.shape = {current_Q1.shape},current_Q2.shape = {current_Q2.shape}')

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, save_dir, name=''):

        torch.save(self.critic.state_dict(), os.path.join(save_dir, "TD3_critic" + '_' + name))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(save_dir,"TD3_critic_optimizer" + '_' + name))

        torch.save(self.actor.state_dict(), os.path.join(save_dir, "TD3_actor" + '_' + name))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(save_dir,"TD3_actor_optimizer" + '_' + name))

    def load(self, save_dir, name=''):
        self.critic.load_state_dict(torch.load(os.path.join(save_dir, "TD3_critic" + '_' + name)))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(save_dir, "TD3_critic_optimizer" + '_' + name)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(os.path.join(save_dir, "TD3_actor" + '_' + name)))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(save_dir,"TD3_actor_optimizer" + '_' + name)))
        self.actor_target = copy.deepcopy(self.actor)

