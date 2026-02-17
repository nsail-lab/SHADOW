from model.PPO import PPOAgent
from model.SequentialTD3 import SequentialTD3
from time import time


class StateThinkingPursuerAgent:
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

        self.movement_agent = SequentialTD3(
            state_dim=state_dim,
            action_dim=action_dim,
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

        self.communication_agent = PPOAgent(
            n_actions=2,
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

        # TD3 for predicting opponent's next state diff
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

        print('DEBUG - actor PPO = ', self.count_params(self.communication_agent.actor))
        print('DEBUG - critic PPO = ', self.count_params(self.communication_agent.critic))
        print('DEBUG - actor TD3 = ', self.count_params(self.movement_agent.actor))
        print('DEBUG - critic TD3 = ', self.count_params(self.movement_agent.critic))
        print('DEBUG - actor Opponent TD3 = ', self.count_params(self.opponent_predictor.actor))
        print('DEBUG - critic Opponent TD3 = ', self.count_params(self.opponent_predictor.critic))


    def count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def sample_action(self, state):
        mov_action = self.movement_agent.sample_action()
        com_action = self.communication_agent.choose_action(state)  # PPO does not require sample action
        pred_dx, pred_dy, pred_dpsi, uncertainty = self.opponent_predictor.sample_action()
        return mov_action, com_action, (pred_dx, pred_dy, pred_dpsi), uncertainty

    def select_train_action(self, state):
        mov_action = self.movement_agent.select_train_action(state)
        com_action = self.communication_agent.choose_action(state)
        pred_dx, pred_dy, pred_dpsi, uncertainty = self.opponent_predictor.select_train_action(state)
        return mov_action, com_action, (pred_dx, pred_dy, pred_dpsi), uncertainty

    def select_action(self, state):
        # Update pursuer and evader agents
        mov_action = self.movement_agent.select_action(state)

        com_action = self.communication_agent.choose_action(state)
        
        pred_dx, pred_dy, pred_dpsi, uncertainty = self.opponent_predictor.select_action(state)
        
        return mov_action, com_action[0], (pred_dx, pred_dy, pred_dpsi), uncertainty

    def train(self):
        self.communication_agent.train()
        self.movement_agent.train()
        self.opponent_predictor.train()

    def store_buffer(self,
                     move_pursuer_actions, next_pursuer_states,  # TD3
                     com_action, probs, vals,  # PPO
                     pred_dx, pred_dy, pred_dpsi, opponent_state_uncertainty, loss, # Opponent TD3
                     pursuer_states, rewards, done,  # both
                     ):
        self.movement_agent.replay_buffer.add(pursuer_states, move_pursuer_actions, rewards, next_pursuer_states, done)
        self.communication_agent.remember(pursuer_states, com_action, probs, vals, rewards, done)
        self.opponent_predictor.replay_buffer.add(pursuer_states, [pred_dx, pred_dy, pred_dpsi, opponent_state_uncertainty], -loss, next_pursuer_states, done)


    def clear_history(self, ):
        self.communication_agent.clear_history()
        self.movement_agent.clear_history()
        self.opponent_predictor.clear_history()

    def save(self, save_dir):
        self.communication_agent.save_models(save_dir)
        self.movement_agent.save(save_dir)
        self.opponent_predictor.save(save_dir, name='opponent_state_diff_predictor')

    def load(self, filename):
        self.communication_agent.load_models(filename)
        self.movement_agent.load(filename)
        self.opponent_predictor.load(filename, name='opponent_state_diff_predictor')
        
