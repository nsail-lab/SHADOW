
from model.TD3_original import TD3Agent
from model.TD3_LSTM import TD3Agent as TD3_LSTM_Agent
from model.PPO import PPOAgent
from model.SequentialTD3 import SequentialTD3
from model.PDQN import P_DQN
import time


class StateThinkingPursuerAgent_PDQN:
    def __init__(self,
                 state_dim,
                 max_action,
                 action_dim,
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
        
        self.opponent_predictor = SequentialTD3(
            state_dim=state_dim,  # Uses the same state dim as movement agent
            action_dim=4,  # Predicting opponent state diff dx, dy, dpsi + uncertainty
            max_action=max_action,
            history_len=history_len,
            hidden_dim=hidden_dim,
            discount=discount,
            tau=tau,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_delay=policy_delay,
            expl_noise=expl_noise,
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            cuda=cuda
        )


        print('DEBUG - actor PDQN = ', self.count_params(self.mov_com_agent.actor))
        print('DEBUG - critic PDQN = ', self.count_params(self.mov_com_agent.critic))

        print('DEBUG - actor Opponent TD3 = ', self.count_params(self.opponent_predictor.actor))
        print('DEBUG - critic Opponent TD3 = ', self.count_params(self.opponent_predictor.critic))


    def clear_history(self):
        self.opponent_predictor.clear_history()
    
    def count_params(self, model):
        return sum(p.numel() for p in model.parameters())
    
    def sample_action(self, state):
        
        com_action, action_params = self.mov_com_agent.select_action(state, train=True)
        mov_action = action_params[0,0]
        
        pred_dx, pred_dy, pred_dpsi, uncertainty = self.opponent_predictor.sample_action()
        
        return (mov_action, 0), (com_action, 0, 0), (pred_dx, pred_dy, pred_dpsi), uncertainty

    def select_train_action(self, state):
        com_action, action_params = self.mov_com_agent.select_action(state, train=True)
        mov_action = action_params[0,0]
        
        pred_dx, pred_dy, pred_dpsi, uncertainty = self.opponent_predictor.select_train_action(state)
        return (mov_action, 0), (com_action, 0, 0), (pred_dx, pred_dy, pred_dpsi), uncertainty
        
    def select_action(self, state):
        com_action, action_params = self.mov_com_agent.select_action(state, train=False)
        mov_action = action_params[0,0]
        
        pred_dx, pred_dy, pred_dpsi, uncertainty = self.opponent_predictor.select_action(state)

        return (mov_action, 0), com_action, (pred_dx, pred_dy, pred_dpsi), uncertainty

    def train(self):
        self.mov_com_agent.update()
        self.opponent_predictor.train()
        
    def store_buffer(self,
                     move_pursuer_actions, next_pursuer_states,  # TD3
                     com_action, probs, vals,  # PPO
                     pred_dx, pred_dy, pred_dpsi, opponent_state_uncertainty, loss, # Opponent TD3
                     pursuer_states, rewards, done,  # both
                     ):    
        self.mov_com_agent.memory.push(pursuer_states, com_action, (move_pursuer_actions,), rewards, next_pursuer_states, done)
        self.opponent_predictor.replay_buffer.add(pursuer_states, [pred_dx, pred_dy, pred_dpsi, opponent_state_uncertainty], -loss, next_pursuer_states, done)

    def save(self, save_dir):
        self.mov_com_agent.save_models(save_dir)
        self.opponent_predictor.save(save_dir, name='opponent_state_diff_predictor')

    def load(self, filename):
        self.mov_com_agent.load_models(filename)
        self.opponent_predictor.load(filename, name='opponent_state_diff_predictor')