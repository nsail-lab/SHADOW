import numpy as np
from mediators.Mediator import Mediator


class SelfSupervisedStateMediator(Mediator):
    """
    Mediator for ThinkingPursuerAgent. Simulates an environment with full knowledge.
    1. Retrieves prediction of opponent state and uncertainty change
    2. Updates prediction of opponents state and its uncertainty accordingly.
    3. Fetches observed state from environment, and pads it with predicted opponents change in state.
    4. When a query is made, replaces prediction with zero state change and sets uncertainty to zero.
    5. Adjusts the reward towards the opponent state diff prediction network.
    """
    def __init__(self, opponent="evader", gamma=0.999, lambd=0.9, epsilon=1e-3):
        super().__init__()
        self.opponent = opponent
        self.map_size = None
        self.dt = None
        self.opponent_speed = None
        self.opponent_max_acceleration = None

        self.last_observed_self = None  # Last observed state of player
        self.last_observed_relative_opponent = None  # Last observed state of opponent relative to self

        self.pursuer_query_action = None  # Current query action
        self.state_self = None  # Current state of self

        self.pred_opponent_state_diff_sequence = None  # Sequence of predicted actions from last observed state
        self.uncertainty_opponent_state = None  # Uncertainty in current opponent relative state

        self.loss = None
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon

    def reset(self, env):
        """When environment is reset, opponent location and game parameters are known."""
        self.map_size = env.map_size
        self.dt = env.dt
        if self.opponent == "evader":
            self.opponent_speed = env.evader_speed
            self.opponent_max_acceleration = env.evader_max_acceleration
            self.last_observed_self = np.array([env.last_record_p_e['x'],
                                                env.last_record_p_e['y'],
                                                env.last_record_p_e['psi']])
            self.last_observed_relative_opponent = np.array([env.last_record_e_p['x'] - env.last_record_p_e['x'],
                                                             env.last_record_e_p['y'] - env.last_record_p_e['y'],
                                                             env._normalize_angle(
                                                                 env.last_record_e_p['psi'] - env.last_record_p_e['psi']
                                                             )])
            self.state_self = np.array([env.pursuer['x'],
                                        env.pursuer['y'],
                                        env.pursuer['psi']])
        elif self.opponent == "pursuer":
            self.opponent_speed = env.pursuer_speed
            self.opponent_max_acceleration = env.pursuer_max_acceleration
            self.last_observed_self = np.array([env.last_record_e_p['x'],
                                                env.last_record_e_p['y'],
                                                env.last_record_e_p['psi']])
            self.last_observed_relative_opponent = np.array([env.last_record_p_e['x'] - env.last_record_e_p['x'],
                                                             env.last_record_p_e['y'] - env.last_record_e_p['y'],
                                                             self._normalize_angle(
                                                                 env.last_record_p_e['psi'] - env.last_record_e_p['psi']
                                                             )])
            self.state_self = np.array([env.evader['x'],
                                        env.evader['y'],
                                        env.evader['psi']])
        else:
            assert False, "Opponent not supported, use either evader or pursuer."

        self.pred_opponent_state_diff_sequence = [np.zeros(self.last_observed_relative_opponent.shape)]
        self.uncertainty_opponent_state = np.array([0.0])

        self.pursuer_query_action = None  # No action was taken yet
        self.loss = 0  # Accumulates prediction errors


    def process_action(self, action):
        """Extracts the action components from the agent and updates the opponent state prediction."""
        _, pursuer_query_action, pred_opponent_state_diff, pred_opponent_uncertainty_delta = action

        # Store query action
        self.pursuer_query_action = pursuer_query_action

        # Store predicted state diff
        self.pred_opponent_state_diff_sequence.append(np.array([*pred_opponent_state_diff, pred_opponent_uncertainty_delta]))

        # Update uncertainty, and clip to ensure it remains non-negative
        self.uncertainty_opponent_state = np.maximum(0, self.uncertainty_opponent_state + pred_opponent_uncertainty_delta)
        
            
        
    def process_observation(self, observation, info=None):
        """
        Modify the agent's observation by adding:
        - Predicted opponent state diff.
        - Uncertainty of that prediction.
        - If query_action is True, set uncertainty to zero and use true opponent state.

        """
        # Compute the loss
        self.loss = None
        if info:
            global_state = info['state']
            opponent_state = global_state[:3] if self.opponent=='pursuer' else global_state[5:8]
            opponent_last_observed = self.last_observed_self + self.last_observed_relative_opponent
            state_diff_label = opponent_state - opponent_last_observed
            dx_pred, dy_pred, dpsi_pred, dsigma_pred = self.pred_opponent_state_diff_sequence[-1]
            state_diff_pred = np.array([dx_pred, dy_pred, dpsi_pred])
            prediction_error = state_diff_label - state_diff_pred
            prediction_error[2] = self._normalize_angle(prediction_error[2]) # angle is defined mod 360°
            n = len(self.pred_opponent_state_diff_sequence) - 1
            # Compute Gaussian NLL loss
            state_diff_loss = self.lambd * (self.gamma**n) * (
                (prediction_error**2).sum() / (2*(self.uncertainty_opponent_state + self.epsilon))
            )
            dsigma_loss = self.lambd * (self.gamma**n) * 0.5 * np.log(self.uncertainty_opponent_state + self.epsilon)

            # Uncertainty regularization loss
            uncertainty_reg_loss = (1-self.lambd) * (self.gamma**n) * max(0, -dsigma_pred)
            self.loss = state_diff_loss + dsigma_loss + uncertainty_reg_loss

        state = observation[0]  # Original state without opponent action history

        if self.pursuer_query_action:        
            self.uncertainty_opponent_state = np.array([0.0])
            self.pred_opponent_state_diff_sequence = [np.zeros(self.last_observed_relative_opponent.shape)]
            # self.loss = 0 # added right now
            
            
        enhanced_state = np.concatenate((state, self.pred_opponent_state_diff_sequence[-1][:3], self.uncertainty_opponent_state))

        return enhanced_state

    def _normalize_angle(self, angle):
        # Normalize angle to be within [-pi, pi]
        return (angle + np.pi) % (2 * np.pi) - np.pi

