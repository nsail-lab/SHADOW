
from model.TD3_original import TD3Agent
from model.TD3_LSTM import TD3Agent as TD3_LSTM_Agent
from model.PPO import PPOAgent
from model.SequentialTD3 import SequentialTD3
import time

from model.ContinuousPPO import PPOContinuousAgent

class EvaderAgent_MHPPO:
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

        self.movement_agent = PPOContinuousAgent(
            input_dims=(state_dim,),
            alpha=alpha,
            gamma = discount,
            gae_lambda=gae_lambda,
            policy_clip=policy_clip,
            history_len=history_len,
            batch_size=batch_size,
            replay_buffer_size = replay_buffer_size,
            cuda = cuda,
        )

        print('DEBUG - actor MHPPO = ', self.count_params(self.movement_agent.actor))
        print('DEBUG - critic MHPPO = ', self.count_params(self.movement_agent.critic))


    def count_params(self, model):
        return sum(p.numel() for p in model.parameters())
    
    def sample_action(self, state):
        action, logprob, value = self.movement_agent.choose_action(state)
        return action, logprob, value


    def select_train_action(self, state):
        action, logprob, value = self.movement_agent.choose_action(state)
        return action, logprob, value

    def select_action(self, state):
        action, logprob, value = self.movement_agent.choose_action(state)
        return action, logprob, value

    def train(self):
        #t_start = time.time()
        self.movement_agent.train()
        #print('DEBUG >>> communication train = ', time.time()-t_start)


    def store_buffer(self, 
                     move_pursuer_actions, next_pursuer_states, # TD3
                     com_action, probs, vals, # PPO
                     pursuer_states, rewards, done, # both
                     ):
        self.movement_agent.remember(pursuer_states, 
                                     move_pursuer_actions, 
                                     probs, 
                                     vals, 
                                     rewards, 
                                     done)

    
    def clear_history(self,):
        self.movement_agent.clear_history()

    def save(self, save_dir):
        self.movement_agent.save(save_dir)

    def load(self, filename):
        self.movement_agent.load(filename)
