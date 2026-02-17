
from model.TD3_original import TD3Agent
from model.TD3_LSTM import TD3Agent as TD3_LSTM_Agent
from model.PPO import PPOAgent
from model.SequentialTD3 import SequentialTD3
import time


class PursuerAgent:
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
                 alpha=0.0003,
                 gae_lambda = 0.95,
                 policy_clip=0.2,
                 history_len = 10,
                 batch_size=256,
                 replay_buffer_size=100000,
                 cuda="0"):
        
        # self.movement_agent = TD3Agent(
        #     state_dim= state_dim,
        #     action_dim= action_dim,
        #     max_action= max_action,
        #     hidden_dim= hidden_dim,
        #     discount= discount,
        #     tau= tau,
        #     policy_noise= policy_noise,
        #     noise_clip= noise_clip,
        #     policy_delay= policy_delay,
        #     expl_noise= expl_noise,
        #     batch_size= batch_size,
        #     replay_buffer_size = replay_buffer_size,
        # )

        # remember to uncomment clear_history 
        self.movement_agent = SequentialTD3(
            state_dim= state_dim,
            action_dim= action_dim,
            max_action= max_action,
            history_len = history_len, 
            hidden_dim= hidden_dim,
            discount= discount,
            tau= tau,
            policy_noise= policy_noise,
            noise_clip= noise_clip,
            policy_delay= policy_delay,
            expl_noise= expl_noise,
            batch_size= batch_size,
            replay_buffer_size = replay_buffer_size,    
            cuda=cuda        
        )

        
        self.communication_agent = PPOAgent(
            n_actions = 2, #2, 
            input_dims= (state_dim,) , 
            gamma= discount, 
            alpha= alpha, 
            gae_lambda= gae_lambda,
            policy_clip= policy_clip, 
            history_len = history_len, 
            batch_size= batch_size, 
            replay_buffer_size = replay_buffer_size,
            cuda=cuda
        )

        print('DEBUG - actor PPO = ', self.count_params(self.communication_agent.actor))
        print('DEBUG - critic PPO = ', self.count_params(self.communication_agent.critic))

        print('DEBUG - actor TD3 = ', self.count_params(self.movement_agent.actor))
        print('DEBUG - critic TD3 = ', self.count_params(self.movement_agent.critic))


    def count_params(self, model):
        return sum(p.numel() for p in model.parameters())
    
    def sample_action(self, state):
        mov_action = self.movement_agent.sample_action()
        # print('[Valerio] DEBUG state.shape ==> ', state.shape)
        com_action = self.communication_agent.choose_action(state) # PPO does not require sample action
        return mov_action,com_action

    def select_train_action(self, state):
        mov_action = self.movement_agent.select_train_action(state)
        com_action = self.communication_agent.choose_action(state)
        return mov_action,com_action

    def select_action(self, state):
        mov_action = self.movement_agent.select_action(state)
        com_action = self.communication_agent.choose_action(state)
        
        return mov_action, com_action[0]

    def train(self):
        #t_start = time.time()
        self.communication_agent.train()
        #print('DEBUG >>> communication train = ', time.time()-t_start)

        #t_start = time.time()
        self.movement_agent.train()
        #print('DEBUG >>> movement train = ', time.time()-t_start)

    def store_buffer(self, 
                     move_pursuer_actions, next_pursuer_states, # TD3
                     com_action, probs, vals, # PPO
                     pursuer_states, rewards, done, # both
                     ):
        
        self.movement_agent.replay_buffer.add(pursuer_states, move_pursuer_actions, rewards, next_pursuer_states, done)
        self.communication_agent.remember(pursuer_states, com_action, probs, vals, rewards, done)
    
    def clear_history(self,):
        self.communication_agent.clear_history()
        self.movement_agent.clear_history()

    def save(self, save_dir):
        self.communication_agent.save_models(save_dir)
        self.movement_agent.save(save_dir)

    def load(self, filename):
        self.communication_agent.load_models(filename)
        self.movement_agent.load(filename)