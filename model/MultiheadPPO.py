import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from collections import deque
import os 

class PPOMemory:
    def __init__(self,  max_size, history_len, cuda="0"):
        self.max_size = max_size
        self.buffer = []
        self.history_len = history_len

        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")


    def sample(self, batch_size):
        buffer_len = len(self.buffer)
        if buffer_len < batch_size:
            # If buffer size is smaller than batch_size, sample with replacement
            batch_idx = np.random.choice(np.arange(buffer_len), batch_size, replace=True)
        else:
            batch_idx = np.random.choice(buffer_len, batch_size, replace=False)


        states_batch = []
        actions_batch = {'comm': [], 'move': []}
        probs_batch = {'comm': [], 'move': []}
        vals_batch = []
        rewards_batch = []
        # next_states_batch = []
        dones_batch = []

        # Construct the batch
        for idx in batch_idx:
            # Ensure the sequence length doesn't exceed history_len
            start_idx = max(0, idx - self.history_len + 1)
            
            # Create a sequence of states by slicing the buffer
            state_sequence = [self.buffer[i][0] for i in range(start_idx, idx + 1)]
            
            # Padding the sequence with the first state (if it's shorter than history_len)
            state_sequence = [state_sequence[0]] * (self.history_len - len(state_sequence)) + state_sequence
            
            states_batch.append(state_sequence)
            
            actions_batch['comm'].append(self.buffer[idx][1]['comm'])
            actions_batch['move'].append(self.buffer[idx][1]['move'])
            
            probs_batch['comm'].append(self.buffer[idx][2]['comm'])
            probs_batch['move'].append(self.buffer[idx][2]['move'])
            
            vals_batch.append(self.buffer[idx][3])
            rewards_batch.append(self.buffer[idx][4])
            # next_states_batch.append(self.buffer[idx][5])
            dones_batch.append(self.buffer[idx][5])

        # Convert to tensors
        states_batch = torch.FloatTensor(np.array(states_batch)).to(self.device)
        
        actions_batch['comm'] = torch.FloatTensor(np.array(actions_batch['comm'])).to(self.device)
        actions_batch['move'] = torch.FloatTensor(np.array(actions_batch['move'])).to(self.device)
        
        probs_batch['comm'] = torch.FloatTensor(np.array(probs_batch['comm'])).to(self.device)
        probs_batch['move'] = torch.FloatTensor(np.array(probs_batch['move'])).to(self.device)
                
        vals_batch = torch.FloatTensor(np.array(vals_batch)).to(self.device)
        rewards_batch = torch.FloatTensor(np.array(rewards_batch).reshape(batch_size, -1)).to(self.device)
        # next_states_batch = torch.FloatTensor(np.array(next_states_batch).reshape(batch_size, -1)).to(self.device)
        dones_batch = torch.FloatTensor(np.array(dones_batch).reshape(batch_size, -1)).to(self.device)

        return states_batch, actions_batch, probs_batch, vals_batch, rewards_batch, dones_batch

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        experience = (state, action, probs, vals, reward, done)
        self.buffer.append(experience)

        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def clear_memory(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)
    

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo', 
                 cuda="0", max_action=1.0):
        super(ActorNetwork, self).__init__()

        # Store max_action for scaling movement output
        self.max_action = max_action

        # Shared encoder (LSTM-based)
        self.lstm = nn.LSTM(*input_dims,fc1_dims,batch_first=True)

        # Communication head (discrete)
        self.comm_fc = nn.Linear(fc1_dims, fc1_dims)
        self.comm_logits = nn.Linear(fc1_dims, 2)  # binary logits

        # Movement head (continuous)
        self.move_fc = nn.Linear(fc1_dims, fc2_dims)
        self.mu_head = nn.Linear(fc2_dims, 1)
        self.log_std = nn.Parameter(torch.zeros(1))  # learnable log std

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # Device
        self.device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state_sequence):
        # state_sequence shape: [batch_size, seq_len, feature_dim]
        lstm_out, (hn, cn) = self.lstm(state_sequence)  # hn: [1, batch, fc1_dims]
        last_hidden = hn[-1]  # shape: [batch, fc1_dims]

        # Communication head (Categorical)
        x_comm = torch.relu(self.comm_fc(last_hidden))
        dist = self.comm_logits(x_comm)
        comm_dist = Categorical(logits=dist)

        # Movement head (Gaussian)
        x_move = torch.relu(self.move_fc(last_hidden))
        mu = self.max_action * torch.tanh(self.mu_head(x_move))  # bounded output
        std = torch.exp(self.log_std)
        move_dist = Normal(mu, std)

        return comm_dist, move_dist

    def save_checkpoint(self, filename):
        torch.save(self.state_dict(), filename)

    def load_checkpoint(self, filename):
        self.load_state_dict(torch.load(filename, map_location=self.device))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, cuda="0"):
        super(CriticNetwork, self).__init__()

        self.lstm = nn.LSTM(*input_dims, fc1_dims, batch_first=True)

        # Feedforward value head
        self.value_head = nn.Sequential(
            nn.Linear(fc1_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, 1)
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # Device setup
        self.device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state_sequence):
        """
        state_sequence: shape (batch_size, sequence_length, feature_dim)
        """
        _, (hn, _) = self.lstm(state_sequence)
        last_hidden = hn[-1]  # shape: (batch_size, hidden_dim)
        value = self.value_head(last_hidden)  # output shape: (batch_size,)
        
        return value

    def save_checkpoint(self, filename):
        torch.save(self.state_dict(), filename)

    def load_checkpoint(self, filename):
        self.load_state_dict(torch.load(filename, map_location=self.device))


class MultiheadPPOAgent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, history_len = 10, batch_size=64, replay_buffer_size = 1_000, cuda="0"):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda

        self.batch_size = batch_size
        self.history_len = history_len

        self.device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

        self.actor = ActorNetwork(n_actions, input_dims, alpha, cuda=cuda)
        self.critic = CriticNetwork(input_dims, alpha,cuda=cuda)
        self.memory = PPOMemory(replay_buffer_size,history_len=history_len,cuda=cuda)

        self.state_history = deque(maxlen=self.history_len)

    def clear_history(self,):
        self.state_history = deque(maxlen=self.history_len)


    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, save_dir):
        #print('... saving models ...')
        # os.path.join(save_dir,'PPO_actor')
        self.actor.save_checkpoint(os.path.join(save_dir,'PPO_actor'))
        self.critic.save_checkpoint(os.path.join(save_dir,'PPO_critic'))

    def load_models(self, load_dir):
        #print('... loading models ...')
        self.actor.load_checkpoint(os.path.join(load_dir,'PPO_actor'))
        self.critic.load_checkpoint(os.path.join(load_dir,'PPO_critic'))

    def update_state_history(self, new_state):
        # Add new state to the history buffer
        self.state_history.append(new_state)
        
        
        # Pad the history buffer if it has fewer than `history_len` entries
        if len(self.state_history) < self.memory.history_len:
            first_state = self.state_history[0]
            padding_states = [first_state] * (self.memory.history_len - len(self.state_history)) # repeat the initial state
            return np.array(padding_states + list(self.state_history))
        else:
            return np.array(list(self.state_history))
        

    def choose_action(self, state, eval_mode=False):
        # print('state ====> ', type(state), state.shape, state)
        
        state_sequence = self.update_state_history(state)
        state_sequence = torch.tensor(state_sequence, dtype=torch.float32).to(self.device)
        # print('state_sequence.shape ===> ', state_sequence.shape)
        actor_output = self.actor(state_sequence)
        dist = actor_output[0]
        movement_dist = actor_output[1]
        
        if not eval_mode:
            action = dist.sample()
            move_action = movement_dist.sample()
        else:
            action = torch.argmax(dist.probs, dim=-1)
            move_action = movement_dist.mean
            
        value = self.critic(state_sequence)
        
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        
        move_probs = torch.squeeze(movement_dist.log_prob(move_action)).item()
        move_action = torch.squeeze(move_action).item()
        
        return action, probs, value, move_action, move_probs

    def train(self):

        # sample buffer
        states_batch, actions_batch, probs_batch, vals_batch, rewards_batch, dones_batch = self.memory.sample(batch_size=self.batch_size)
        # print('DEBUG - states_batch.shape, rewards_batch.shape, vals_batch.shape', states_batch.shape, rewards_batch.shape, vals_batch.shape)

        rewards_batch_normalised = (rewards_batch-rewards_batch.mean())/(rewards_batch.std()+ 1e-8)

        # advantages computation
        advantages = torch.zeros(len(rewards_batch), dtype=torch.float32, device=self.actor.device)
        # print(['DEBUG - advantage.shape', advantage.shape])

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

        advantages = (advantages-advantages.mean())/(advantages.std()+ 1e-8)
        returns = advantages + vals_batch
        
        dist, mov_action = self.actor(states_batch)
        
        critic_value = self.critic(states_batch)
        critic_value = torch.squeeze(critic_value)
        # dist = self.actor(states_batch)

        comm_actions = actions_batch['comm']       
        move_actions = actions_batch['move']      

        # new_probs = dist.log_prob(comm_actions)
        # Extract old log probs
        old_comm_log_probs = probs_batch['comm']
        old_move_log_probs = probs_batch['move']   
        
        # Compute new log probs
        new_comm_log_probs = dist.log_prob(comm_actions)
        new_move_log_probs = mov_action.log_prob(move_actions).sum(dim=-1)  # if multi-dimensional

        # Compute ratios
        comm_ratio = torch.exp(new_comm_log_probs - old_comm_log_probs)
        move_ratio = torch.exp(new_move_log_probs - old_move_log_probs)

        # PPO clipping objective per head
        comm_weighted = comm_ratio * advantages
        comm_clipped = torch.clamp(comm_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
        comm_loss = -torch.min(comm_weighted, comm_clipped).mean()

        move_weighted = move_ratio * advantages
        move_clipped = torch.clamp(move_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
        move_loss = -torch.min(move_weighted, move_clipped).mean()

        
        critic_loss = (returns-critic_value)**2
        critic_loss = critic_loss.mean()

        # Total loss
        total_loss = comm_loss + move_loss + 0.5 * critic_loss

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()