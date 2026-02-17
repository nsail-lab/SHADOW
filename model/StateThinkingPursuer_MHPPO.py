from model.PPO import PPOAgent
from model.MultiheadPPO import MultiheadPPOAgent
from model.SequentialTD3 import SequentialTD3
from time import time


class StateThinkingPursuerAgent_MHPPO:
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

        print('DEBUG - actor MHPPO = ', self.count_params(self.movement_comm_agent.actor))
        print('DEBUG - critic MHPPO = ', self.count_params(self.movement_comm_agent.critic))

        print('DEBUG - actor Opponent TD3 = ', self.count_params(self.opponent_predictor.actor))
        print('DEBUG - critic Opponent TD3 = ', self.count_params(self.opponent_predictor.critic))


    def count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def sample_action(self, state):
        action, probs, value, move_action, move_probs = self.movement_comm_agent.choose_action(state, eval_mode=False)
        pred_dx, pred_dy, pred_dpsi, uncertainty = self.opponent_predictor.sample_action()
        return (move_action, move_probs), (action, probs, value), (pred_dx, pred_dy, pred_dpsi), uncertainty

    def select_train_action(self, state):
        action, probs, value, move_action, move_probs = self.movement_comm_agent.choose_action(state, eval_mode=False)
        
        pred_dx, pred_dy, pred_dpsi, uncertainty = self.opponent_predictor.select_train_action(state)
        return (move_action, move_probs), (action, probs, value), (pred_dx, pred_dy, pred_dpsi), uncertainty

    def select_action(self, state):
        action, probs, value, move_action, move_probs = self.movement_comm_agent.choose_action(state, eval_mode=True)
        
        pred_dx, pred_dy, pred_dpsi, uncertainty = self.opponent_predictor.select_action(state)

        return (move_action, move_probs), action, (pred_dx, pred_dy, pred_dpsi), uncertainty

    def train(self):
        self.movement_comm_agent.train()
        self.opponent_predictor.train()

    def store_buffer(self,
                     move_pursuer_actions, next_pursuer_states,  # TD3
                     com_action, probs, vals,  # PPO
                     pred_dx, pred_dy, pred_dpsi, opponent_state_uncertainty, loss, # Opponent TD3
                     pursuer_states, rewards, done,  # both
                     ):
        self.movement_comm_agent.remember(pursuer_states,
                                            action = {'comm': com_action, 'move': move_pursuer_actions},
                                            probs = {'comm': probs, 'move': 0},
                                            vals=vals,
                                            reward=rewards,
                                            done=done)
        
        # self.movement_agent.replay_buffer.add(pursuer_states, move_pursuer_actions, rewards, next_pursuer_states, done)
        # self.communication_agent.remember(pursuer_states, com_action, probs, vals, rewards, done)
        self.opponent_predictor.replay_buffer.add(pursuer_states, [pred_dx, pred_dy, pred_dpsi, opponent_state_uncertainty], -loss, next_pursuer_states, done)


    def clear_history(self, ):
        self.movement_comm_agent.clear_history()
        self.opponent_predictor.clear_history()

    def save(self, save_dir):
        self.movement_comm_agent.save_models(save_dir)
        self.opponent_predictor.save(save_dir, name='opponent_state_diff_predictor')

    def load(self, filename):
        self.movement_comm_agent.load_models(filename)
        self.opponent_predictor.load(filename, name='opponent_state_diff_predictor')