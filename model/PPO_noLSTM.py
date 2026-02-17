import os
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from collections import deque



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
        actions_batch = []
        probs_batch = []
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
            actions_batch.append(self.buffer[idx][1])
            probs_batch.append(self.buffer[idx][2])
            vals_batch.append(self.buffer[idx][3])
            rewards_batch.append(self.buffer[idx][4])
            # next_states_batch.append(self.buffer[idx][5])
            dones_batch.append(self.buffer[idx][5])

        # Convert to tensors
        states_batch = torch.FloatTensor(np.array(states_batch)).to(self.device)
        actions_batch = torch.FloatTensor(np.array(actions_batch)).to(self.device)
        probs_batch = torch.FloatTensor(np.array(probs_batch)).to(self.device)
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
        self.experience = []

    def __len__(self):
        return len(self.buffer)
    
    
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo', cuda = "0"):
        super(ActorNetwork, self).__init__()

        #self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        
        # self.lstm = nn.LSTM(*input_dims, fc1_dims, batch_first=True)
        self.layer1 = nn.Linear(*input_dims, fc1_dims)
        #self.layer2 = nn.Linear(fc1_dims, fc2_dims)
        self.layer3 = nn.Linear(fc1_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state_sequence):
        
        # lstm_out, (hn, cn) = self.lstm(state_sequence)
        x = torch.relu(self.layer1(state_sequence))
        #x = torch.relu(self.layer2(x))
        dist = torch.softmax(self.layer3(x),dim=-1)

        dist = Categorical(dist.squeeze(-1))
        return dist

    def save_checkpoint(self, filename):
        torch.save(self.state_dict(), filename)

    def load_checkpoint(self, filename):
        self.load_state_dict(torch.load(filename, map_location=self.device))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo',cuda="0"):
        super(CriticNetwork, self).__init__()

        #self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        # self.lstm = nn.LSTM(*input_dims, fc1_dims, batch_first=True)
        self.layer1 = nn.Linear(*input_dims, fc1_dims)
        #self.layer2 = nn.Linear(fc1_dims, fc2_dims)
        self.layer3 = nn.Linear(fc1_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state_sequence):

        # lstm_out, (hn, cn) = self.lstm(state_sequence)
        x = torch.relu(self.layer1(state_sequence))
        #x = torch.relu(self.layer2(x))
        value = torch.relu(self.layer3(x))

        return value.squeeze(-1)

    def save_checkpoint(self,filename):
        torch.save(self.state_dict(), filename)

    def load_checkpoint(self, filename):
        self.load_state_dict(torch.load(filename, map_location=self.device))

class PPOAgent:
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
        

    def choose_action(self, state):
        state_sequence = self.update_state_history(state)
        state_sequence = torch.tensor(state_sequence, dtype=torch.float32).to(self.device)
        # print('DEBUG >> state_sequence.shape', state_sequence.shape)
        dist = self.actor(state_sequence)
        value = self.critic(state_sequence)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

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

        dist = self.actor(states_batch)
        
        critic_value = self.critic(states_batch)
        critic_value = torch.squeeze(critic_value)

        new_probs = dist.log_prob(actions_batch)
        
        prob_ratio = new_probs.exp() / probs_batch.exp()
        #prob_ratio = (new_probs - old_probs).exp()
        weighted_probs = advantages * prob_ratio
        weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                1+self.policy_clip)*advantages
        actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

        returns = advantages + vals_batch
        critic_loss = (returns-critic_value)**2
        critic_loss = critic_loss.mean()

        total_loss = actor_loss + 0.5*critic_loss
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()