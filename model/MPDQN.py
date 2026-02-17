import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os 

# ---------------------------
# Replay Buffer
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity, cuda = '0'):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device(f'cuda:{cuda}')

    def push(self, state, discrete_action, cont_action, reward, next_state, done):
        self.buffer.append((state, discrete_action, cont_action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            # Sample with replacement using numpy
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=True)
        else:
            # Sample without replacement
            indices = random.sample(range(len(self.buffer)), batch_size)

        batch = [self.buffer[i] for i in indices]
        
        state, d_a, c_a, r, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(self.device),
            torch.LongTensor(d_a).to(self.device),
            torch.FloatTensor(c_a).to(self.device),
            torch.FloatTensor(r).unsqueeze(1).to(self.device),
            torch.FloatTensor(next_state).to(self.device),
            torch.FloatTensor(done).unsqueeze(1).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)

# ---------------------------
# MP-DQN Network Components
# ---------------------------
class StateEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, state):
        return self.fc(state)

class ContinuousActor(nn.Module):
    def __init__(self, state_dim, discrete_actions, cont_action_dim, hidden_dim):
        super().__init__()
        self.discrete_actions = discrete_actions
        self.cont_action_dim = cont_action_dim
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, cont_action_dim),
            ) for _ in range(discrete_actions)
        ])

    def forward(self, state):
        # Output shape: (batch_size, discrete_actions, cont_action_dim)
        return torch.stack([net(state) for net in self.fc], dim=1)

class QNetwork(nn.Module):
    def __init__(self, hidden_dim, discrete_actions, cont_action_dim):
        super().__init__()
        self.q_fc = nn.Sequential(
            nn.Linear(hidden_dim + cont_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.discrete_actions = discrete_actions

    def forward(self, encoded_state, cont_actions):
        # cont_actions shape: (batch_size, discrete_actions, cont_action_dim)
        # encoded_state shape: (batch_size, hidden_dim)
        batch_size = encoded_state.size(0)
        encoded_state_exp = encoded_state.unsqueeze(1).expand(-1, self.discrete_actions, -1)
        q_input = torch.cat([encoded_state_exp, cont_actions], dim=-1)
        q_values = self.q_fc(q_input)  # (batch_size, discrete_actions, 1)
        return q_values.squeeze(-1)    # (batch_size, discrete_actions)

# ---------------------------
# MP-DQN Agent
# ---------------------------
class MPDQNAgent:
    def __init__(self, state_dim, discrete_actions, cont_action_dim, hidden_dim=256,
                 gamma=0.99, tau=0.005, lr=1e-3, buffer_size=1000000, batch_size=64, cuda = '0'):
        self.device = torch.device(f'cuda:{cuda}')
        self.discrete_actions = discrete_actions
        self.cont_action_dim = cont_action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.encoder = StateEncoder(state_dim, hidden_dim).to(self.device)
        self.actor = ContinuousActor(state_dim, discrete_actions, cont_action_dim, hidden_dim).to(self.device)
        self.qnet = QNetwork(hidden_dim, discrete_actions, cont_action_dim).to(self.device)

        self.encoder_target = StateEncoder(state_dim, hidden_dim).to(self.device)
        self.actor_target = ContinuousActor(state_dim, discrete_actions, cont_action_dim, hidden_dim).to(self.device)
        self.qnet_target = QNetwork(hidden_dim, discrete_actions, cont_action_dim).to(self.device)

        self._hard_update(self.encoder_target, self.encoder)
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.qnet_target, self.qnet)

        self.optim_encoder = optim.Adam(self.encoder.parameters(), lr=lr)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optim_qnet = optim.Adam(self.qnet.parameters(), lr=lr)

        self.buffer = ReplayBuffer(capacity=buffer_size,cuda=cuda)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        cont_actions = self.actor(state)  # (1, discrete_actions, cont_action_dim)
        encoded = self.encoder(state)     # (1, hidden_dim)
        q_vals = self.qnet(encoded, cont_actions)  # (1, discrete_actions)
        action = torch.argmax(q_vals, dim=1).item()
        cont = cont_actions[0, action].detach().cpu().numpy()
        return action, cont

    def push(self, *args):
        self.buffer.push(*args)

    def update(self):

        state, d_a, c_a, reward, next_state, done = self.buffer.sample(self.batch_size)
        state, d_a, c_a, reward, next_state, done = [
            x.to(self.device) for x in (state, d_a, c_a, reward, next_state, done)
        ]

        # -- Critic update --
        with torch.no_grad():
            next_cont = self.actor_target(next_state)
            next_encoded = self.encoder_target(next_state)
            next_q = self.qnet_target(next_encoded, next_cont)
            next_q_max = next_q.max(1, keepdim=True)[0]
            q_target = reward + self.gamma * (1 - done) * next_q_max

        encoded = self.encoder(state)
        q_pred = self.qnet(encoded, self.actor(state))
        q_taken = q_pred.gather(1, d_a.unsqueeze(1))

        q_loss = F.mse_loss(q_taken, q_target)

        self.optim_qnet.zero_grad()
        self.optim_encoder.zero_grad()
        q_loss.backward()
        self.optim_qnet.step()
        self.optim_encoder.step()

        # -- Actor update --
        cont_pred = self.actor(state)
        q_val = self.qnet(self.encoder(state), cont_pred)
        actor_loss = -q_val.mean()

        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        # -- Target soft update --
        self._soft_update(self.encoder_target, self.encoder)
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.qnet_target, self.qnet)

    def _soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(self.tau * s_param.data + (1 - self.tau) * t_param.data)

    def _hard_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(s_param.data)
            
    def save_models(self, save_dir, name=''):
        torch.save(self.encoder.state_dict(), os.path.join(save_dir, f'MPDQN_encoder_{name}'))
        torch.save(self.actor.state_dict(), os.path.join(save_dir, f'MPDQN_actor_{name}'))
        torch.save(self.qnet.state_dict(), os.path.join(save_dir, f'MPDQN_qnet_{name}'))

        torch.save(self.encoder_target.state_dict(), os.path.join(save_dir, f'MPDQN_encoder_target_{name}'))
        torch.save(self.actor_target.state_dict(), os.path.join(save_dir, f'MPDQN_actor_target_{name}'))
        torch.save(self.qnet_target.state_dict(), os.path.join(save_dir, f'MPDQN_qnet_target_{name}'))       
        
        print('Models saved successfully')

    def load_models(self, save_dir, name=''):
        device = self.device
        self.encoder.load_state_dict(torch.load(os.path.join(save_dir, f'MPDQN_encoder_{name}'), map_location=device))
        self.actor.load_state_dict(torch.load(os.path.join(save_dir, f'MPDQN_actor_{name}'), map_location=device))
        self.qnet.load_state_dict(torch.load(os.path.join(save_dir, f'MPDQN_qnet_{name}'), map_location=device))

        self.encoder_target.load_state_dict(torch.load(os.path.join(save_dir, f'MPDQN_encoder_target_{name}'), map_location=device))
        self.actor_target.load_state_dict(torch.load(os.path.join(save_dir, f'MPDQN_actor_target_{name}'), map_location=device))
        self.qnet_target.load_state_dict(torch.load(os.path.join(save_dir, f'MPDQN_qnet_target_{name}'), map_location=device))

        print('Models loaded successfully')