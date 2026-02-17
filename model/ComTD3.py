import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Link to TD3 model
# https://github.com/sfujim/TD3/blob/master/TD3.py


class ReplayBuffer:
    def __init__(self, max_size, device):
        self.max_size = max_size
        self.buffer = []
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done, hidden, cell, action_log_prob):
        experience = (state, action, reward, next_state, done, hidden, cell, action_log_prob)
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
        states, actions, rewards, next_states, dones, hiddens, cells, action_log_probs = zip(*batch)

        states = torch.FloatTensor(np.array(states).reshape(batch_size, -1)).to(self.device)
        actions = torch.FloatTensor(np.array(actions).reshape(batch_size, -1)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(batch_size, -1)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states).reshape(batch_size, -1)).to(self.device)
        dones = torch.FloatTensor(np.array(dones).reshape(batch_size, -1)).to(self.device)
        hiddens = torch.FloatTensor(np.array(hiddens).reshape(batch_size, -1)).to(self.device)
        cells = torch.FloatTensor(np.array(hiddens).reshape(batch_size, -1)).to(self.device)
        action_log_probs = torch.FloatTensor(np.array(action_log_probs).reshape(batch_size, -1)).to(self.device)

        return states, actions, rewards, next_states, dones, hiddens, cells, action_log_probs

# we now need to "remember" what has happened before, the "Memory" allows the agent to do this
# The representation learnt by the LSTM is the input to the following layers.
# This should help the agent to minimise the number of times, it queries (i.e, communicates) the state.
class MemoryUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(MemoryUnit, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x, h):
        x, h = self.lstm(x.unsqueeze(1), (h[0].unsqueeze(0), h[1].unsqueeze(0)))
        return x, h


# TODO: use Actor & Critic from TD3 (set state_dim=lstm_hidden_dim)
class Actor(nn.Module):
    def __init__(self, hidden_dim, action_dim, max_action): # h_dim instead of input_dim
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.max_action * torch.tanh(self.layer3(x))
        return x


class Critic(nn.Module):
    def __init__(self, hidden_dim, action_dim): # h_dim instead of input_dim
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# binary prob. output, communicate/non-communicate
class PPOPolicy(nn.Module):
    def __init__(self, hidden_dim, action_dim): # h_dim instead of input_dim
        super(PPOPolicy, self).__init__()
        # Policy network
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)

        # Value network (for V(s))
        self.value_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer3 = nn.Linear(hidden_dim, 1)  # Outputs a single value for V(s)

    def forward(self, x):
        # Policy network forward pass
        policy_x = torch.relu(self.layer1(x))
        policy_x = torch.relu(self.layer2(policy_x))
        action_probs = torch.softmax(self.layer3(policy_x), dim=-1)

        # Value network forward pass
        value_x = torch.relu(self.value_layer1(x))
        value_x = torch.relu(self.value_layer2(value_x))
        state_value = self.value_layer3(value_x)
        
        return action_probs, state_value


class ComTD3Agent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 query_dim,
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
                 ppo_update_frequency=1,
                 ppo_clip_param=0.1,
                 cuda="0"):
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.device)
        self.shared_memory_unit = MemoryUnit(state_dim, hidden_dim).to(self.device)
        self.shared_memory_unit_optimizer = optim.Adam(self.shared_memory_unit.parameters(), lr=3e-6)

        self.actor = Actor(hidden_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(hidden_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1.5e-6)

        self.critic1 = Critic(hidden_dim, action_dim).to(self.device)
        self.critic1_target = Critic(hidden_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-6)

        self.critic2 = Critic(hidden_dim, action_dim).to(self.device)
        self.critic2_target = Critic(hidden_dim, action_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-6)

        self.ppo_policy = PPOPolicy(hidden_dim, query_dim).to(self.device)
        self.ppo_optimizer = optim.Adam(self.ppo_policy.parameters(), lr=1.5e-6)

        self.max_action = max_action
        self.action_dim = action_dim
        self.query_dim = query_dim
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.expl_noise = expl_noise
        self.total_it = 0
        self.batch_size = batch_size
        self.ppo_update_frequency = ppo_update_frequency
        self.ppo_clip_param = ppo_clip_param

        # Initialize the hidden state for the memory unit
        self.hidden = torch.zeros(1, 1, hidden_dim).to(self.device)  # h0: (num_layers, batch_size, hidden_dim)
        self.cell = torch.zeros(1, 1, hidden_dim).to(self.device)  # c0: (num_layers, batch_size, hidden_dim)
        # Store the last action log probability
        self.last_action_log_prob = None

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        hidden = torch.FloatTensor(self.get_hidden_state())
        cell = torch.FloatTensor(self.get_cell_state())
        output, (self.hidden, self.cell) = self.shared_memory_unit(state, (hidden, cell))
        return self.actor(output).cpu().data.numpy().flatten()

    def query_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        hidden = torch.FloatTensor(self.get_hidden_state())
        cell = torch.FloatTensor(self.get_cell_state())
        output, (self.hidden, self.cell) = self.shared_memory_unit(state, (hidden, cell))
        # Compute action probabilities
        action_probs, _ = self.ppo_policy(output)
        action_probs = action_probs.reshape((-1,))

        # Sample the action and compute the log probability
        action = torch.multinomial(action_probs.reshape((1, -1)), 1)
        log_prob = torch.log(action_probs)

        # Store the last action log probability
        self.last_action_log_prob = log_prob

        return action.cpu().data.numpy().flatten() 

    def sample_action(self):
        return npr.uniform(low=-self.max_action, high=self.max_action, size=self.action_dim)

    def sample_query(self):
        random_actions = npr.uniform(size=self.query_dim)
        random_action_probs = torch.softmax(torch.FloatTensor(random_actions).to(self.device), dim=-1)
        random_action = np.array([npr.choice(self.query_dim, p=random_action_probs.cpu().detach().numpy())])
        random_log_prob = torch.log(random_action_probs + 1e-8)

        # Store the last action log probability
        self.last_action_log_prob = random_log_prob

        random_action = [1.]
        self.last_action_log_prob = torch.FloatTensor([-2.3026, -0.1054]).to(self.device)
        return random_action

    def select_train_action(self, state):
        return (self.select_action(state) +
                npr.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                ).clip(-self.max_action, self.max_action)

    def select_train_query(self, state):
        return (self.query_action(state) +
                npr.normal(0, 0.5 * self.expl_noise, size=1)
                ).clip(0, 1) 

    def get_hidden_state(self):
        return self.hidden.reshape((1, -1)).cpu().detach().numpy()

    def get_cell_state(self):
        return self.cell.reshape((1, -1)).cpu().detach().numpy()

    def reset_memory(self):
        # Reset hidden state, usually at the beginning of an episode
        self.hidden = torch.zeros(self.hidden.size()).to(self.device)
        self.cell = torch.zeros(self.hidden.size()).to(self.device)

    def get_action_log_prob(self):
        # Return the log probability of the last action
        return self.last_action_log_prob.cpu().detach().numpy()

    def train(self):
        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, done, hidden, cell, old_action_log_prob = self.replay_buffer.sample(self.batch_size)

        # Convert to torch tensor
        state = torch.FloatTensor(state.reshape((self.batch_size, -1))).to(self.device)
        next_state = torch.FloatTensor(next_state.reshape((self.batch_size, -1))).to(self.device)
        hidden = torch.FloatTensor(hidden.reshape((self.batch_size, -1))).to(self.device)
        cell = torch.FloatTensor(cell.reshape((self.batch_size, -1))).to(self.device)

        # Forward pass through the shared LSTM for the current state
        output, (hidden, cell) = self.shared_memory_unit(state, (hidden, cell))
        hidden = hidden.reshape((self.batch_size, -1))
        cell = cell.reshape((self.batch_size, -1))
        output = output.reshape((self.batch_size, -1))

        # Forward pass through the shared LSTM for the next state
        next_output, _ = self.shared_memory_unit(next_state, (hidden, cell))
        next_output = next_output.reshape((self.batch_size, -1))

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_output) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_q1 = self.critic1_target(next_output, next_action)
            target_q2 = self.critic2_target(next_output, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1-done) * self.discount * target_q

        # Get current Q estimates
        current_q1 = self.critic1(output, action)
        current_q2 = self.critic2(output, action)

        # Compute critic loss
        critic1_loss = nn.functional.mse_loss(current_q1, target_q)
        critic2_loss = nn.functional.mse_loss(current_q2, target_q)
        # Combine the critic losses into one
        total_critic_loss = critic1_loss + critic2_loss

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic1(output, self.actor(output)).mean()
            total_loss = total_critic_loss + actor_loss

            # Update the frozen target models
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            total_loss = total_critic_loss

        # # PPO Update 
        # # the training of this PPO is independent w.r.t. the other training
        # if self.total_it % self.ppo_update_frequency == 0:
        #     # Compute PPO loss
        #     action_probs, state_value = self.ppo_policy(output)

        #     action_log_probs = torch.log(action_probs + 1e-8)  # Added for numerical stability
        #     dist_entropy = -torch.sum(action_probs * action_log_probs, dim=-1).mean()

        #     ratios = torch.exp(action_log_probs - old_action_log_prob)

        #     _, next_state_value = self.ppo_policy(next_output)
        #     # Compute the TD target
        #     with torch.no_grad():
        #         td_target = reward + (1 - done) * self.discount * next_state_value
        #     advantages = td_target - state_value

        #     # Normalize advantages
        #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        #     surr1 = ratios * advantages
        #     surr2 = torch.clamp(ratios, 1.0 - self.ppo_clip_param, 1.0 + self.ppo_clip_param) * advantages
        #     ppo_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy

        #     total_loss += ppo_loss

        # Optimize the critic
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        # Optimize the actor
        if self.total_it % self.policy_delay == 0:
            self.actor_optimizer.zero_grad()
        # Optimize the LSTM (since it is part of the critic calculation)
        self.shared_memory_unit_optimizer.zero_grad()
        # Optimize the PPO policy
        # if self.total_it % self.ppo_update_frequency == 0:
        #     self.ppo_optimizer.zero_grad()

        total_loss.backward()

        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        if self.total_it % self.policy_delay == 0:
            self.actor_optimizer.step()
        # if self.total_it % self.ppo_update_frequency == 0:
        #     self.ppo_optimizer.step(d)
        self.shared_memory_unit_optimizer.step()

    def save(self, filename):
        torch.save(self.shared_memory_unit.state_dict(), filename + "_shared_memory_unit")
        torch.save(self.shared_memory_unit_optimizer.state_dict(), filename + "_shared_memory_unit_optimizer")

        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")

        torch.save(self.critic2.state_dict(), filename + "_critic2")
        torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.shared_memory_unit.load_state_dict(torch.load(filename + "_shared_memory_unit"))
        self.shared_memory_unit_optimizer.load_state_dict(torch.load(filename + "_shared_memory_unit_optimizer"))

        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer"))
        self.critic1_target = copy.deepcopy(self.critic1)

        self.critic2.load_state_dict(torch.load(filename + "_critic2"))
        self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer"))
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


