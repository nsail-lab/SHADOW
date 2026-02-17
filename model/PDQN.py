import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import random
import gym.spaces as spaces
import math
import torch.nn.functional as F
import os

# @author Metro
# @time 2021/11/3

"""
  Ref: https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/net.py
       https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.distributions import Normal, TransformedDistribution, TanhTransform

"""
Source: https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg.py
        https://github.com/cycraig/MP-DQN/blob/master/agents/memory/memory.py
"""
import numpy as np
import random


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def clear(self):
        self.start = 0
        self.length = 0
        self.data[:] = 0  # unnecessary, not freeing any memory, could be slow


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, observation_shape, action_shape, next_actions=False):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.terminals = RingBuffer(limit, shape=(1,))

    def sample(self, batch_size, random_machine=np.random):
        batch_idxs = random_machine.random_integers(low=0, high=self.nb_entries - 1, size=batch_size)

        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)

        if next_actions is not None:
            return states_batch, actions_batch, rewards_batch, next_states_batch, next_actions, terminals_batch
        else:
            return states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch

    def append(self, state, action, reward, next_state, next_action=None, terminal=False, training=True):
        if not training:
            return

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions:
            self.next_actions.append(next_action)
        self.terminals.append(terminal)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.next_actions.clear()
        self.terminals.clear()

    @property
    def nb_entries(self):
        return len(self.states)


class MemoryV2(object):
    def __init__(self, limit, observation_shape, action_shape, next_actions=False, time_steps=False):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.time_steps = RingBuffer(limit, shape=(1,)) if time_steps else None
        self.terminals = RingBuffer(limit, shape=(1,))

    def sample(self, batch_size, random_machine=np.random):
        # Draw such that we always have a proceeding element.
        # batch_idxs = random_machine.random_integers(self.nb_entries - 2, size=batch_size)
        batch_idxs = random_machine.choice(self.nb_entries, size=batch_size)
        # batch_idxs = random_machine.choice(self.nb_entries, weights=[i/self.nb_entries for i in range(
        # self.nb_entries)], size=batch_size)

        '''states_batch = array_min2d(self.states.get_batch(batch_idxs))
        actions_batch = array_min2d(self.actions.get_batch(batch_idxs))
        rewards_batch = array_min2d(self.rewards.get_batch(batch_idxs))
        next_states_batch = array_min2d(self.next_states.get_batch(batch_idxs))
        terminals_batch = array_min2d(self.terminals.get_batch(batch_idxs))'''
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)
        time_steps = self.time_steps.get_batch(batch_idxs) if self.time_steps is not None else None

        ret = [states_batch, actions_batch, rewards_batch, next_states_batch]
        if next_actions is not None:
            ret.append(next_actions)
        ret.append(terminals_batch)
        if time_steps is not None:
            ret.append(time_steps)
        return tuple(ret)

    def append(self, state, action, reward, next_state, next_action=None, terminal=False, time_steps=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions is not None:
            self.next_actions.append(next_action)
        self.terminals.append(terminal)
        if self.time_steps is not None:
            self.time_steps.append(time_steps)

    @property
    def nb_entries(self):
        return len(self.states)


class MemoryNStepReturns(object):
    def __init__(self, limit, observation_shape, action_shape, next_actions=False, time_steps=False,
                 n_step_returns=False):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.time_steps = RingBuffer(limit, shape=(1,)) if time_steps else None
        self.terminals = RingBuffer(limit, shape=(1,))
        self.n_step_returns = RingBuffer(limit, shape=(1,)) if n_step_returns else None

    def sample(self, batch_size, random_machine=np.random):
        # Draw such that we always have a proceeding element.
        # batch_idxs = random_machine.random_integers(self.nb_entries - 2, size=batch_size)
        batch_idxs = random_machine.choice(self.nb_entries, size=batch_size)
        # batch_idxs = random_machine.choice(self.nb_entries, weights=[i/self.nb_entries for i in range(
        # self.nb_entries)], size=batch_size)

        '''states_batch = array_min2d(self.states.get_batch(batch_idxs))
        actions_batch = array_min2d(self.actions.get_batch(batch_idxs))
        rewards_batch = array_min2d(self.rewards.get_batch(batch_idxs))
        next_states_batch = array_min2d(self.next_states.get_batch(batch_idxs))
        terminals_batch = array_min2d(self.terminals.get_batch(batch_idxs))'''
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)
        time_steps = self.time_steps.get_batch(batch_idxs) if self.time_steps is not None else None
        n_step_returns = self.n_step_returns.get_batch(batch_idxs) if self.n_step_returns is not None else None

        ret = [states_batch, actions_batch, rewards_batch, next_states_batch]
        if next_actions is not None:
            ret.append(next_actions)
        ret.append(terminals_batch)
        if time_steps is not None:
            ret.append(time_steps)
        if n_step_returns is not None:
            ret.append(n_step_returns)
        return tuple(ret)

    def append(self, state, action, reward, next_state, next_action=None, terminal=False, time_steps=None,
               n_step_return=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions is not None:
            self.next_actions.append(next_action)
        self.terminals.append(terminal)
        if self.time_steps is not None:
            assert time_steps is not None
            self.time_steps.append(time_steps)
        if self.n_step_returns is not None:
            assert n_step_return is not None
            self.n_step_returns.append(n_step_return)

    @property
    def nb_entries(self):
        return len(self.states)


class ReplayBuffer:
    def __init__(self, capacity=1e5):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, action_param, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, action_param, reward, next_state, done

    def push(self, state, action, action_param, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, action_param, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def __len__(self):
        return len(self.buffer)


def init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


epsilon = 1e-6


class DuelingDQN(nn.Module):

    def __init__(self, state_dim, continuous_action_dim, discrete_action_dim, hidden_layers=(256, 128, 64),
                 ):
        """

        :param state_dim:
        :param action_dim:
        :param hidden_layers:
        """
        super().__init__()

        # initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim + continuous_action_dim, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

        self.adv_layers_1 = nn.Linear(hidden_layers[-1], discrete_action_dim)
        self.val_layers_1 = nn.Linear(hidden_layers[-1], 1)

        self.adv_layers_2 = nn.Linear(hidden_layers[-1], discrete_action_dim)
        self.val_layers_2 = nn.Linear(hidden_layers[-1], 1)

        self.apply(init_)

    def forward(self, state, action_params):
        temp = torch.cat((state, action_params), dim=1)

        x1 = temp
        for i in range(len(self.layers)):
            x1 = F.relu(self.layers[i](x1))
        adv1 = self.adv_layers_1(x1)
        val1 = self.val_layers_1(x1)
        q_duel1 = val1 + adv1 - adv1.mean(dim=1, keepdim=True)

        x2 = temp
        for i in range(len(self.layers)):
            x2 = F.relu(self.layers[i](x2))
        adv2 = self.adv_layers_1(x2)
        val2 = self.val_layers_1(x2)
        q_duel2 = val2 + adv2 - adv2.mean(dim=1, keepdim=True)

        return q_duel1, q_duel2


class GaussianPolicy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_layers=(256, 128, 64), action_space=None,cuda='0',
                 ):
        """

        :param state_dim:
        :param action_dim:
        :param hidden_layers:
        """
        super().__init__()
        self.cuda = cuda
        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")

        # initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        self.mean_layers = nn.Linear(hidden_layers[-1], action_dim)
        self.log_std_layers = nn.Linear(hidden_layers[-1], action_dim)

        # action rescaling
        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1., device=self.device)
            self.action_bias = torch.tensor(0., device=self.device)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.).to(self.device)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.).to(self.device)


        self.apply(init_)

    def forward(self, state):
        x = state.to(self.device)

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))
        mean = self.mean_layers(x)
        log_std = self.log_std_layers(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state):
        state = state.to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        tanh_normal = TransformedDistribution(normal, [TanhTransform(cache_size=1)])

        x_t = tanh_normal.rsample()
        action = x_t * self.action_scale + self.action_bias

        log_prob = tanh_normal.log_prob(x_t)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # sum over action dims

        return action, log_prob, mean


class P_DQN(object):
    """
    A soft actor-critic agent for hybrid action spaces

    """

    NAME = 'P-DQN Agent'

    def __init__(self, state_dim, 
                 max_continous_action, cuda='0'):
        self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        
        self.hyperparameters = {
            'device': f"cuda:{cuda}" if torch.cuda.is_available() else 'cpu',
            'epsilon_initial': 0.3,
            'epsilon_final': 0.01,
            'epsilon_decay': 5000,
            'replay_memory_size': 1e6,
            'batch_size': 64,
            'gamma': 0.99,
            'lr_critic': 1e-5,
            'lr_actor': 1e-4,
            'lr_alpha': 1e-2,
            'tau_actor': 0.01,
            'tau_critic': 0.01,
            'critic_hidden_layers': (256, 128, 64),
            'actor_hidden_layers': (256, 128, 64),
            'random_pick_steps': 10000,
            'updates_per_step': 2,
            'maximum_episodes': 2000,
            'alpha': 0.2,
        }

        self.state_dim = state_dim
        print(f"State dimension: {self.state_dim}")
        
        self.discrete_action_dim = 2  # this will be 2
        self.continuous_action_dim = 1  # this will be 1
        
        self.action_space = spaces.Tuple((
            spaces.Discrete(2),
            spaces.Box(low=-max_continous_action, high=max_continous_action,shape=(1,), dtype=np.float32)
        ))        
        
        self.epsilon = self.hyperparameters['epsilon_initial']
        self.epsilon_initial = self.hyperparameters['epsilon_initial']
        self.epsilon_final = self.hyperparameters['epsilon_final']
        self.epsilon_decay = self.hyperparameters['epsilon_decay']
        self.batch_size = self.hyperparameters['batch_size']
        self.gamma = self.hyperparameters['gamma']

        self.lr_critic = self.hyperparameters['lr_critic']
        self.lr_actor = self.hyperparameters['lr_actor']
        self.lr_alpha = self.hyperparameters['lr_alpha']
        self.tau_critic = self.hyperparameters['tau_critic']
        self.tau_actor = self.hyperparameters['tau_actor']
        self.critic_hidden_layers = self.hyperparameters['critic_hidden_layers']
        self.actor_hidden_layers = self.hyperparameters['actor_hidden_layers']

        self.counts = 0
        self.alpha = 0.2

        # ----  Initialization  ----
        self.memory = ReplayBuffer(capacity=self.hyperparameters['replay_memory_size'])
        
        self.critic = DuelingDQN(self.state_dim, self.continuous_action_dim, self.discrete_action_dim, self.critic_hidden_layers).to(self.device)
        self.critic_target = DuelingDQN(self.state_dim, self.continuous_action_dim, self.discrete_action_dim, self.critic_hidden_layers).to(self.device)

        # self.critic = DuelingDQN(self.state_dim, self.continuous_action_dim, self.critic_hidden_layers,
        #                          ).to(self.device)
        # self.critic_target = DuelingDQN(self.state_dim, self.continuous_action_dim, self.critic_hidden_layers,
        #                                 ).to(self.device)
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.hard_update(source=self.critic, target=self.critic_target)

        self.actor = GaussianPolicy(
            self.state_dim, self.continuous_action_dim, self.actor_hidden_layers, self.action_space[1],cuda=cuda).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        self.target_entropy = -torch.Tensor([self.continuous_action_dim]).to(self.device).item()
        self.log_alpha = torch.tensor(-np.log(self.continuous_action_dim), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_critic)  # todo

    def hard_update(self,target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
            
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def select_action(self, state, train=True):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        self.epsilon = self.epsilon_final + (self.epsilon_initial - self.epsilon_final) * math.exp(-1. * self.counts / self.epsilon_decay)
        self.counts += 1
        if train:
            with torch.no_grad():
                # state = torch.FloatTensor(state).to(self.device)
                # print(type(state), state)
                action_params, _, _ = self.actor.sample(state)
                # print('action_params:', action_params.shape)
                if random.random() < self.epsilon:
                    action = np.random.randint(2)

                else:
                    Q_a, _ = self.critic(state, action_params)
                    # print('Q_a:', Q_a)
                    
                    Q_a = Q_a.detach().cpu().numpy()
                    action = int(np.argmax(Q_a))
                
                
                
                               
                action_params = action_params.detach().cpu().numpy()
        else:
            with torch.no_grad():
                _, _, action_params = self.actor.sample(state)
                # print('action_params:', action_params.shape)
                
                Q_a, _ = self.critic.forward(state, action_params)
                # print('Q_a:', Q_a)
                # print('Q_a:', Q_a)
                
                
                Q_a = Q_a.detach().cpu().numpy()
                action = int(np.argmax(Q_a))
                action_params = action_params.detach().cpu().numpy()

        return action, action_params

    def update(self, ):
        state_batch, action_batch, action_params_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.IntTensor(action_batch).to(self.device).long().unsqueeze(1)
        action_params_batch = torch.FloatTensor(action_params_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        
        # print(f"State batch shape: {state_batch.shape}")
        # print(f"Next state batch shape: {next_state_batch.shape}")
        # print(f"Action batch shape: {action_batch.shape}")
        # print(f"Action params batch shape: {action_params_batch.shape}")
        # print(f"Reward batch shape: {reward_batch.shape}")
        # print(f"Done batch shape: {done_batch.shape}")
        
        # ------------------------------------ update critic -----------------------------------------------
        with torch.no_grad():
            next_state_action_params, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_state_action_params)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
            q_next = reward_batch + (1 - done_batch) * self.gamma * min_q_next_target
        q1, q2 = self.critic(state_batch, action_params_batch)
        q_loss = F.mse_loss(q1, q_next) + F.mse_loss(q2, q_next)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()
        self.soft_update(self.critic_target, self.critic, self.tau_critic)

        # ------------------------------------ update actor -----------------------------------------------
        pi, log_pi, _ = self.actor.sample(state_batch)
        q1_pi, q2_pi = self.critic(state_batch, pi)
        # min_q_pi = torch.min(q1_pi.gather(1, action_batch), q2_pi.gather(1, action_batch))
        min_q_pi = torch.min(q1_pi.mean(), q2_pi.mean())

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------------------ update alpha -----------------------------------------------
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.detach().exp()
        
    def save_models(self, save_dir, name='', ):
        
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'PDQN_critic_' + name))
        
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'PDQN_actor_' + name))
        print('Models saved successfully')

    def load_models(self, save_dir, name='',):
        # also try load on CPU if no GPU available?
        
        self.critic.load_state_dict(torch.load(os.path.join(save_dir, 'PDQN_critic_' + name)))
        self.actor.load_state_dict(torch.load(os.path.join(save_dir, 'PDQN_actor_' + name)))
        print('Models loaded successfully')
