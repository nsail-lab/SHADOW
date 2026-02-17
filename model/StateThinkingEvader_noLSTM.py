from model.PPO_noLSTM import PPOAgent
from model.SequentialTD3 import SequentialTD3
from model.TD3_original import TD3Agent
from time import time


class StateThinkingEvaderAgent:
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
                 gae_lambda=0.95,
                 policy_clip=0.2,
                 history_len=10,
                 batch_size=256,
                 replay_buffer_size=100000,
                 cuda="0"):

        self.movement_agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
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

        # TD3 for predicting opponent's next state diff
        self.opponent_predictor = TD3Agent(
            state_dim=state_dim,  # Uses the same state dim as movement agent
            action_dim=4,  # Predicting opponent state diff dx, dy, dpsi + uncertainty
            max_action=max_action,
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

        print('DEBUG - actor TD3 = ', self.count_params(self.movement_agent.actor))
        print('DEBUG - critic TD3 = ', self.count_params(self.movement_agent.critic))
        print('DEBUG - actor Opponent TD3 = ', self.count_params(self.opponent_predictor.actor))
        print('DEBUG - critic Opponent TD3 = ', self.count_params(self.opponent_predictor.critic))


    def count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def sample_action(self, state):
        mov_action = self.movement_agent.sample_action()
        pred_dx, pred_dy, pred_dpsi, uncertainty = self.opponent_predictor.sample_action()
        return mov_action, (pred_dx, pred_dy, pred_dpsi), uncertainty

    def select_train_action(self, state):
        mov_action = self.movement_agent.select_train_action(state)
        pred_dx, pred_dy, pred_dpsi, uncertainty = self.opponent_predictor.select_train_action(state)
        return mov_action, (pred_dx, pred_dy, pred_dpsi), uncertainty

    def select_action(self, state):
        mov_action = self.movement_agent.select_action(state)
        pred_dx, pred_dy, pred_dpsi, uncertainty = self.opponent_predictor.select_action(state)

        return mov_action, (pred_dx, pred_dy, pred_dpsi), uncertainty

    def train(self):
        self.movement_agent.train()
        self.opponent_predictor.train()

    def store_buffer(self,
                     move_evader_actions, next_evader_states,  # TD3
                     pred_dx, pred_dy, pred_dpsi, opponent_state_uncertainty, loss, # Opponent TD3
                     evader_states, rewards, done,  # both
                     ):
        self.movement_agent.replay_buffer.add(evader_states, move_evader_actions, rewards, next_evader_states, done)
        self.opponent_predictor.replay_buffer.add(evader_states, [pred_dx, pred_dy, pred_dpsi, opponent_state_uncertainty], -loss, next_evader_states, done)


    def clear_history(self, ):
        self.movement_agent.clear_history()
        self.opponent_predictor.clear_history()

    def save(self, save_dir):
        self.movement_agent.save(save_dir)
        self.opponent_predictor.save(save_dir, name='opponent_state_diff_predictor')

    def load(self, filename):
        self.movement_agent.load(filename)
        self.opponent_predictor.load(filename, name='opponent_state_diff_predictor')