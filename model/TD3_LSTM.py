
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.functional as F
from collections import deque


# Link to TD3 model
# https://github.com/sfujim/TD3/blob/master/TD3.py

class ReplayBuffer:
    def __init__(self, max_size, history_len, device):
        self.max_size = max_size
        self.buffer = []
        self.history_len = history_len
        self.device = device

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

        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
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
            rewards_batch.append(self.buffer[idx][2])
            next_states_batch.append(self.buffer[idx][3])
            dones_batch.append(self.buffer[idx][4])

        # Convert to tensors
        states_batch = torch.FloatTensor(np.array(states_batch)).to(self.device)
        actions_batch = torch.FloatTensor(np.array(actions_batch)).to(self.device)
        rewards_batch = torch.FloatTensor(np.array(rewards_batch).reshape(batch_size, -1)).to(self.device)
        next_states_batch = torch.FloatTensor(np.array(next_states_batch).reshape(batch_size, -1)).to(self.device)
        dones_batch = torch.FloatTensor(np.array(dones_batch).reshape(batch_size, -1)).to(self.device)

        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch
    

    def __len__(self):
        return len(self.buffer)

class StateEncoder(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(StateEncoder, self).__init__()
        # LSTM to process history of states
        self.lstm = nn.LSTM(state_dim, hidden_size, batch_first=True)
        
    def forward(self, state_sequence):
        # state_sequence shape: (batch_size, history_len, state_dim)
        lstm_out, (hn, cn) = self.lstm(state_sequence)
        # Return the final hidden state (embedding)
        return hn[-1] # torch.squeeze(state_sequence)  


class Actor(nn.Module):
    def __init__(self, embedding_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.max_action * torch.tanh(self.layer3(x))
        return x

class Critic(nn.Module):
    def __init__(self, embedding_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(embedding_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(embedding_dim + action_dim, hidden_dim)
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
                 embedding_dim = 256,
                 hidden_dim=256,
                 history_len = 1,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 expl_noise=0.1,
                 batch_size=256,
                 replay_buffer_size=100000,
                 cuda="0"):
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ReplayBuffer(replay_buffer_size, history_len=history_len, device=self.device)
        self.state_history = deque(maxlen=history_len)
        self.memory = StateEncoder(state_dim, hidden_size=hidden_dim).to(self.device)
        self.memory_target = StateEncoder(state_dim, hidden_size=hidden_dim).to(self.device)
        self.memory_target.load_state_dict(self.memory.state_dict())

        self.actor = Actor(embedding_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.actor_target = Actor(embedding_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1.5e-6)

        self.critic = Critic(embedding_dim, action_dim).to(self.device)
        self.critic_target = Critic(embedding_dim, action_dim).to(self.device)
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

    def encode_state(self, state_sequence):
        # Pass the state sequence through the StateEncoder to get the embedding
        return self.memory(state_sequence)
    
    def update_state_history(self, new_state):
        # Add new state to the history buffer
        self.state_history.append(new_state)
        
        # Pad the history buffer if it has fewer than `history_len` entries
        if len(self.state_history) < self.replay_buffer.history_len:
            first_state = self.state_history[0]
            padding_states = [first_state] * (self.replay_buffer.history_len - len(self.state_history)) # repeat the initial state
            return np.array(padding_states + list(self.state_history))
        else:
            return np.array(list(self.state_history))
        
    def sample_action(self):
        return npr.uniform(low=-self.max_action, high=self.max_action, size=self.action_dim)
    
    def select_action(self, state):
        # Update state history with the latest state
        state_sequence = self.update_state_history(state)
        
        # Convert state_sequence to tensor and encode it
        with torch.no_grad():
            state_sequence = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
            encoded_state = self.memory(state_sequence)
            action = self.actor(encoded_state).cpu().data.numpy().flatten() 
        return action


    def select_train_action(self, state):
        return (self.select_action(state) +
                np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                ).clip(-self.max_action, self.max_action)
    

    def train(self):
        self.total_it += 1
        # print(f'[DEBUG] total_it = {self.total_it}, policy_delay = {self.policy_delay}')

        # Sample replay buffer
        state_sequence, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        # print(f'DEBUG >>> state_sequence.shape = {state_sequence.shape}, action.shape = {action.shape}, next_state.shape = {next_state.shape}, reward.shape = {reward.shape}, done.shape = {done.shape}')

        reward = (reward-reward.mean())/(reward.std()+ 1e-8)

        with torch.no_grad():

            next_state = next_state.unsqueeze(1) # need to unsqueeze the next state as it is only state (not a sequence)
            encoded_next_state_target = self.memory_target(next_state)
            # print(f'DEBUG >>> next_state.shape = {next_state.shape}, encoded_next_state.shape = {encoded_next_state_target.shape}')


            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(encoded_next_state_target) + noise).clamp(-self.max_action, self.max_action)
            # print(f'DEBUG >>> next_action.shape = {next_action.shape}')

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(encoded_next_state_target, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1-done) * self.discount * target_Q
            # print(f'DEBUG >>> target_Q1.shape = {target_Q1.shape},target_Q2.shape = {target_Q2.shape}, target_Q.shape = {target_Q.shape}, reward.shape = {reward.shape}')
        
        
        # Encode the sequences using the StateEncoder
        encoded_state = self.memory(state_sequence)
        # print(f'DEBUG >>> state_sequence.shape = {state_sequence.shape}, encoded_state.shape = {encoded_state.shape}')

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(encoded_state, action)
        # print(f'DEBUG >>> current_Q1.shape = {current_Q1.shape},current_Q2.shape = {current_Q2.shape}')

        # Compute critic loss
        # print('[DEBUG] computing critic_loss')
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        retain_graph= True if self.total_it % self.policy_delay == 0 else False
        # print('[DEBUG] runninng critic_loss.backward()')
        critic_loss.backward(retain_graph=retain_graph)
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            
            # Compute actor loss
            # print('[DEBUG] computing actor_loss')
            actor_loss = -self.critic.Q1(encoded_state, self.actor(encoded_state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            # print('[DEBUG] runninng actor_loss.backward()')
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.memory.parameters(), self.memory_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)      

    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


