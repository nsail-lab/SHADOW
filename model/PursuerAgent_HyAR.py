
from model.TD3_original import TD3Agent
from model.TD3_LSTM import TD3Agent as TD3_LSTM_Agent
from model.PPO import PPOAgent
from model.SequentialTD3 import SequentialTD3
from model.PDQN import P_DQN
from model.HyAR import HyARAgent
import time
import numpy as np

class PursuerAgent_HyAR:
    def __init__(self,
                 state_dim,
                 action_dim=0,
                 max_action=0,
                 hidden_dim=256,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 expl_noise=0.1,
                 alpha=0.0003,
                 gae_lambda = 0.95,
                 policy_clip=0.2,
                 history_len = 10,
                 batch_size=256,
                 replay_buffer_size=100000,
                 cuda="0"):
        
        
        binary_action_dim = 2  # binary action, 2 choices (0 or 1)
        continuous_action_dim = 1  # continuous action, 1-dimensional (real value)
        embed_dim = 256
        latent_dim = 64
        hidden_dim = 256
        lr = 3e-4
        gamma = 0.99
        self.mov_com_agent = HyARAgent(state_dim, binary_action_dim, continuous_action_dim,
                                        embed_dim=embed_dim, latent_dim=latent_dim, hidden_dim=hidden_dim,
                                        lr=lr, gamma=gamma, cuda=cuda)


        print('DEBUG - actor HyAR = ', self.count_params(self.mov_com_agent.actor))
        print('DEBUG - critic HyAR = ', self.count_params(self.mov_com_agent.critic))



    def clear_history(self):
        pass
    
    def count_params(self, model):
        return sum(p.numel() for p in model.parameters())
    
    def sample_action(self, state):
        
        com_action, action_params = self.mov_com_agent.choose_action(state)
        mov_action = action_params[0]
        
        return mov_action,com_action

    def select_train_action(self, state):
        com_action, action_params = self.mov_com_agent.choose_action(state)
        mov_action = action_params[0]
        
        return mov_action,com_action

    def select_action(self, state):
        com_action, action_params = self.mov_com_agent.choose_action(state)
        mov_action = action_params[0]
        
        return mov_action,com_action

    def train(self):
        self.mov_com_agent.update()

    def store_buffer(self, 
                     move_pursuer_actions, next_pursuer_states, # TD3
                     com_action, probs, vals, # PPO
                     pursuer_states, rewards, done, # both
                     ):
        
        # print(move_pursuer_actions)
        # print('DEBUG - pursuer_states - ', pursuer_states)
        # print('DEBUG - com_action - ', com_action)
        # print('DEBUG - np.array(move_pursuer_actions) - ', np.array(move_pursuer_actions))
        # print('DEBUG - rewards - ', rewards)
        # print('DEBUG - next_pursuer_states - ', next_pursuer_states)
        
        self.mov_com_agent.remember(pursuer_states, com_action, np.array([move_pursuer_actions]), rewards, next_pursuer_states, done)
        
        

    def save(self, save_dir):
        self.mov_com_agent.save_models(save_dir)

    def load(self, filename):
        self.mov_com_agent.load_models(filename)