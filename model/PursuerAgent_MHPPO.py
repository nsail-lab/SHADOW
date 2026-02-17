
from model.TD3_original import TD3Agent
from model.TD3_LSTM import TD3Agent as TD3_LSTM_Agent
from model.PPO import PPOAgent
from model.SequentialTD3 import SequentialTD3
import time
from model.MultiheadPPO import MultiheadPPOAgent


class PursuerAgent_MHPPO:
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
        self.movement_comm_agent = MultiheadPPOAgent(
            n_actions = 1,
            input_dims=(state_dim,),
            gamma=discount,
            alpha=alpha,
            gae_lambda=gae_lambda,
            policy_clip=policy_clip,
            history_len=history_len,
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            cuda=cuda
        )

        print('DEBUG - actor MHPPO = ', self.count_params(self.movement_comm_agent.actor))
        print('DEBUG - critic MHPPO = ', self.count_params(self.movement_comm_agent.critic))


    def count_params(self, model):
        return sum(p.numel() for p in model.parameters())
    
    def sample_action(self, state):
        action, probs, value, move_action, move_probs = self.movement_comm_agent.choose_action(state, eval_mode=False)
        return (move_action, move_probs), (action, probs, value)


    def select_train_action(self, state):
        action, probs, value, move_action, move_probs = self.movement_comm_agent.choose_action(state, eval_mode=False)
        return (move_action, move_probs), (action, probs, value)

    def select_action(self, state):
        action, probs, value, move_action, move_probs = self.movement_comm_agent.choose_action(state, eval_mode=True)

        return (move_action, move_probs), action

    def train(self):
        #t_start = time.time()
        self.movement_comm_agent.train()
        #print('DEBUG >>> communication train = ', time.time()-t_start)


    def store_buffer(self, 
                     move_pursuer_actions, next_pursuer_states, # TD3
                     com_action, probs, vals, # PPO
                     pursuer_states, rewards, done, # both
                     ):
        
        self.movement_comm_agent.remember(pursuer_states,
                                          action = {'comm': com_action, 'move': move_pursuer_actions},
                                          probs = {'comm': probs, 'move': 0},
                                          vals=vals,
                                          reward=rewards,
                                          done=done)
    
    def clear_history(self,):
        self.movement_comm_agent.clear_history()

    def save(self, save_dir):
        self.movement_comm_agent.save_models(save_dir)

    def load(self, filename):
        self.movement_comm_agent.load_models(filename)
