
from model.TD3_original import TD3Agent
from model.TD3_LSTM import TD3Agent as TD3_LSTM_Agent
from model.PPO import PPOAgent
from model.SequentialTD3 import SequentialTD3
from model.PDQN import P_DQN
import time


class PursuerAgent_PDQN:
    def __init__(self,
                 state_dim,
                 max_action,
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
        
        self.mov_com_agent = P_DQN(state_dim,max_action,cuda)
        

        print('DEBUG - actor PDQN = ', self.count_params(self.mov_com_agent.actor))
        print('DEBUG - critic PDQN = ', self.count_params(self.mov_com_agent.critic))



    def clear_history(self):
        pass
    
    def count_params(self, model):
        return sum(p.numel() for p in model.parameters())
    
    def sample_action(self, state):
        
        com_action, action_params = self.mov_com_agent.select_action(state, train=True)
        mov_action = action_params[0,0]
        
        return mov_action,com_action

    def select_train_action(self, state):
        com_action, action_params = self.mov_com_agent.select_action(state, train=True)
        mov_action = action_params[0,0]
        
        return mov_action,com_action

    def select_action(self, state):
        com_action, action_params = self.mov_com_agent.select_action(state, train=False)
        mov_action = action_params[0,0]
        
        return mov_action, com_action

    def train(self):
        self.mov_com_agent.update()

    def store_buffer(self, 
                     move_pursuer_actions, next_pursuer_states, # TD3
                     com_action, probs, vals, # PPO
                     pursuer_states, rewards, done, # both
                     ):
        
        self.mov_com_agent.memory.push(pursuer_states, com_action, (move_pursuer_actions,), rewards, next_pursuer_states, done)
        

    def save(self, save_dir):
        self.mov_com_agent.save_models(save_dir)

    def load(self, filename):
        self.mov_com_agent.load_models(filename)