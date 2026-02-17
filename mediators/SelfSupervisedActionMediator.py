import numpy as np
from mediators.Mediator import Mediator


class SelfSupervisedActionMediator(Mediator):
    """
    Mediator for ThinkingPursuerAgent. Simulates an environment with full knowledge.
    1. Retrieves prediction of opponent action and uncertainty change
    2. Updates prediction of opponents state and its uncertainty accordingly.
    3. Fetches observed state from environment, and pads it with predicted opponents change in state.
    4. When a query is made, replaces prediction with actual state and sets uncertainty to zero.
    5. Adjusts the reward towards the opponent prediction network.
    """
    def __init__(self, opponent="evader", gamma=0.99, lambda_1=1.0, lambda_2=0.1, epsilon=1e-6):
        super().__init__()
        self.opponent = opponent
        self.map_size = None
        self.dt = None
        self.opponent_speed = None
        self.opponent_max_acceleration = None

        self.last_observed_self = None  # Last observed state of player
        self.last_observed_relative_opponent = None  # Last observed state of opponent relative to self

        self.query_action = None  # Current query action
        self.state_self = None  # Current state of self

        self.pred_opponent_state_diff = None # Estimated opponent state diff from last observation
        self.pred_opponent_action_sequence = None  # Sequence of predicted actions from last observed state
        self.uncertainty_opponent_state = None  # Uncertainty in current opponent relative state

        self.loss = None
        self.gamma = gamma
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
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
                                                             env._normalize_angle(
                                                                 env.last_record_p_e['psi'] - env.last_record_e_p['psi']
                                                             )])
            self.state_self = np.array([env.evader['x'],
                                        env.evader['y'],
                                        env.evader['psi']])
        else:
            assert False, "Opponent not supported, use either evader or pursuer."

        self.pred_opponent_state_diff = np.zeros(self.last_observed_relative_opponent.shape)
        self.pred_opponent_action_sequence = []
        self.uncertainty_opponent_state = np.array([0.0])

        self.query_action = None  # No action was taken yet
        self.loss = 0  # Accumulates prediction errors


    def process_action(self, action):
        """Extracts the action components from the agent and updates the opponent state prediction."""
        _, query_action, pred_opponent_action, pred_opponent_uncertainty_delta = action

        # Store query action
        self.query_action = query_action

        # Store predicted action
        self.pred_opponent_action_sequence.append(np.array([pred_opponent_action, pred_opponent_uncertainty_delta]))

        # Compute current estimation of opponent state
        # E = P_tilde + (E_tilde - P_tilde) + (E - E_tilde)
        opponent_estimated_state = (self.last_observed_self
                                    + self.last_observed_relative_opponent
                                    + self.pred_opponent_state_diff)

        # Update predicted state diff
        u = pred_opponent_action
        u = np.clip(u, -self.opponent_max_acceleration, self.opponent_max_acceleration)
        # psi += u * dt
        opponent_estimated_state[2] += u * self.dt  # / self.opponent_speed
        opponent_estimated_state[2] = self._normalize_angle(opponent_estimated_state[2])  # 2\pi rotation diff == no rotation
        opponent_estimated_state[0] += self.opponent_speed * np.cos(opponent_estimated_state[2])
        opponent_estimated_state[1] += self.opponent_speed * np.sin(opponent_estimated_state[2])
        # Keep pursuer within bounds
        opponent_estimated_state[0] = np.clip(opponent_estimated_state[0], 0, self.map_size[0])
        opponent_estimated_state[1] = np.clip(opponent_estimated_state[1], 0, self.map_size[1])

        self.pred_opponent_state_diff = (opponent_estimated_state
                                         - self.last_observed_relative_opponent
                                         - self.last_observed_self)
        self.pred_opponent_state_diff[2] = self._normalize_angle(self.pred_opponent_state_diff[2])

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
            opponent_action_label = info['actions'][0][self.opponent=='evader']
            n = len(self.pred_opponent_action_sequence) - 1
            a_pred, dsigma_pred = self.pred_opponent_action_sequence[-1]
            # Compute Gaussian NLL loss
            action_loss = self.lambda_1 * (self.gamma**n) * (
                ((opponent_action_label -  a_pred)**2 / (2*(self.uncertainty_opponent_state + self.epsilon)))
            )
            dsigma_loss = self.lambda_1 * (self.gamma**n) * 0.5 * np.log(self.uncertainty_opponent_state + self.epsilon)

            # Uncertainty regularization loss
            uncertainty_reg_loss = self.lambda_2 * (self.gamma**n) * max(0, -dsigma_pred)
            self.loss = action_loss + dsigma_loss + uncertainty_reg_loss

        state = observation[0]  # Original state without opponent action history
        if self.query_action:
            self.pred_opponent_state_diff = np.zeros(self.pred_opponent_state_diff.shape)
            self.uncertainty_opponent_state = np.array([0.0])
        enhanced_state = np.concatenate((state, self.pred_opponent_state_diff, self.uncertainty_opponent_state))

        return enhanced_state

    def _normalize_angle(self, angle):
        # Normalize angle to be within [-pi, pi]
        return (angle + np.pi) % (2 * np.pi) - np.pi

