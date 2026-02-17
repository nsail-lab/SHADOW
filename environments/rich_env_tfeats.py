from typing import Any

import numpy as np
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import copy
import random


class DifferentialGameEnvironment(gym.Env):
    """Pursuit-evasion differential signaling game environment.
    The pursuer can decide to query the state, which discloses its own state to the evader.
    This environment is considered `rich' as it also exposes the history of actions taken by the opponent.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, num_pursuers,
                 map_size,
                 pursuer_speed,
                 evader_speed,
                 pursuer_max_acceleration,
                 evader_max_acceleration,
                 r_min,
                 r_shoot,
                 dt=0.01,
                 communication_penalty=0,
                 time_penalty=0,
                 hit_boundary_penalty=0,
                 lat_acceleration_penalty=0,
                 shooting_penalty=100,
                 distance_penalty=100,
                 max_steps=10_000,
                 beta=0.05,
                 probabilistic=True,
                 lambd=2,
                 generalization=True,
                 random_communication=False,
                 random_communication_th=0,
                 communication_period=0,
                 noise_std=0.0):
        self.noise_std = noise_std
        self.dt = dt
        self.max_steps = max_steps
        self.step_counter = 0
        self.num_pursuers = num_pursuers
        self.map_size = map_size
        self.pursuer_speed = pursuer_speed
        self.evader_speed = evader_speed
        self.pursuer_max_acceleration = pursuer_max_acceleration
        self.evader_max_acceleration = evader_max_acceleration
        self.r_min = r_min
        self.r_shoot = r_shoot
        self.beta = beta
        self.communication_penalty = communication_penalty
        self.time_penalty = time_penalty
        self.hit_boundary_penalty = hit_boundary_penalty
        self.lat_acceleration_penalty = lat_acceleration_penalty
        self.distance_penalty = distance_penalty
        self.shooting_penalty = shooting_penalty

        self.init_p_speed = self.pursuer_speed
        self.init_e_speed = self.evader_speed
        self.init_p_acc = self.pursuer_max_acceleration
        self.init_e_acc = self.evader_max_acceleration

        self.probabilistic = probabilistic
        self.lambd = lambd

        self.generalization = generalization

        self.pursuer = {}
        self.evader = {}
        self.history = []

        self.evader_actions = []  # Stores evader actions between queries
        self.pursuer_actions = []  # Stores pursuer actions between queries
        self.last_query_step = 0  # Track last query step

        self.random_communication = random_communication  
        self.random_communication_th = random_communication_th
        self.communication_period = communication_period
        
        # Action space: a numeric value for lateral acceleration (P),
        #               a numeric value for lateral acceleration (E)
        #               a binary value for communication (P),
        self.action_space = spaces.Tuple((spaces.Box(low=-self.pursuer_max_acceleration,
                                                     high=self.pursuer_max_acceleration,
                                                     shape=(1,), dtype=np.float64),
                                          spaces.Box(low=-self.evader_max_acceleration,
                                                     high=self.evader_max_acceleration,
                                                     shape=(1,), dtype=np.float64),
                                          spaces.Discrete(2)),  # Querying action
                                         )

        # Observation space:
        #   State:
        #       Position pursuer (x_p, y_p, psi_p, v_p, a_p)
        #       Relative position evader (x_e-x_p, y_e-y_p, psi_e-psi_p, v_e-v_p, a_e-a_p)
        #       Distance between pursuer and evader, r_min and r_shoot
        #   Sequence of previous actions for pursuer and evader
        self.observation_space = spaces.Tuple((
            spaces.Box(low=np.array([0.0, 0.0, -np.pi, 0.0, 0.0, # pursuer
                                      -self.map_size[0], -self.map_size[1], -np.pi, -np.inf, -np.inf, # evader
                                      0.0, 0.0, 0.0, 0, 0]), # distance, r_min, r_shoot, T_last_comm, T_since_last_comm
                       high=np.array([self.map_size[0], self.map_size[1], np.pi, np.inf, np.pi, # pursuer
                                      self.map_size[0], self.map_size[1], np.pi, np.inf, np.pi, # evader
                                      np.inf, np.inf, np.inf, np.inf, np.inf]), # distance, r_min, r_shoot, T_last_comm, T_since_last_comm
                        dtype=np.float64),
            spaces.Sequence(spaces.Box(low=-self.pursuer_max_acceleration, high=self.pursuer_max_acceleration)), # Sequence of pursuer actions
            spaces.Sequence(spaces.Box(low=-self.evader_max_acceleration, high=self.evader_max_acceleration)) # Sequence of evader actions
        ))

        self.reset()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            np.random.seed(seed=seed)

        # Initialize step counter
        self.step_counter = 0

        self.evader_actions = []
        self.pursuer_actions = []
        self.last_query_step = 0
        
        # Initialize pursuers
        self.pursuer = {'x': np.random.uniform(0., self.map_size[0]),  # x coordinate
                        'y': np.random.uniform(0., self.map_size[1]),  # y coordinate
                        'psi': np.random.uniform(-np.pi, np.pi)}  # heading angle

        # Initialize evader
        self.evader = {'x': np.random.uniform(0., self.map_size[0]),  # x coordinate
                       'y': np.random.uniform(0., self.map_size[1]),  # y coordinate
                       'psi': np.random.uniform(-np.pi, np.pi)}  # heading angle


        if self.generalization:
            speed_ref = np.random.uniform(self.init_p_speed, # / 2,
                                          self.init_p_speed * 2)
            # acc_ref = np.random.uniform(self.init_p_acc / 2, self.init_p_acc * 2)
            # speed_ratio = 2**np.random.uniform(-3, +3)  # v_e/v_p = 2**U[-3,+3]
            # acc_ratio = 2**np.random.uniform(-3, + 3)  # v_e/v_p = 2**U[-3,+3]
            #speed_ratio = np.random.uniform(0.5, 1.5)
            speed_ratio = np.random.uniform(0.1, 4)
            acc_ratio = 1  # TODO: add variance, e.g. np.random.uniform(0.5, 1.5)
            
            self.pursuer_speed = speed_ref
            self.evader_speed = speed_ratio * speed_ref
            # print(self.pursuer_speed, self.evader_speed, speed_ratio, self.evader_speed/self.pursuer_speed)
            
            self.lambd = np.random.uniform(0.5, 4)
            
            # self.pursuer_max_acceleration = acc_ref
            # self.evader_max_acceleration = acc_ratio * acc_ref

        else:
            self.evader_speed = self.init_e_speed
            self.pursuer_speed = self.init_p_speed

        # Pursuer - last recorded position of the evader
        self.last_record_e_p = self.evader.copy()
        # Evader - last recorded position of pursuer
        self.last_record_p_e = self.pursuer.copy()

        # print(f'v_e/v_p={self.evader_speed/self.pursuer_speed}|a_e/a_p={self.evader_max_acceleration/self.pursuer_max_acceleration}')

        self.empty_history()
        state = self._get_state()
        self.record_history(state)
        return state, {}

    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            x_p, y_p, psi_p, x_e, y_e, psi_e, dist = self._get_state()
            fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=150)
            ax.scatter(x_p, y_p, s=30, color='blue')
            ax.scatter(x_e, y_e, s=30, color='red', marker='*')
            ax.grid()
            plt.show()
        else:
            return self._get_state()

    def compute_distance(self, a1, a2):
        return np.sqrt((a1['x'] - a2['x']) ** 2 + (a1['y'] - a2['y']) ** 2)

    def step(self, actions):
        self.step_counter += 1
        
        (pursuer_action, evader_action), query = actions

        if self.random_communication:
            rnd = np.random.uniform(0, 1)
            if rnd < self.random_communication_th:
                query = 1
            else:
                query = 0
                
            if self.random_communication_th == -1:
                c_dist = self.compute_distance(self.pursuer, self.evader)
                p_comm = self.exponential_cdf(c_dist)
                if (rnd < (1-p_comm)) & (self.probabilistic):
                    query = 1
                else:
                    query = 0
            if self.random_communication_th == -2:
                ## Periodical communication strategy as defined in A Pursuit-Evasion Differential Game with Strategic Information Acquisition
                if self.step_counter % self.communication_period == 0:
                    query = 1
                    # print(f'{self.step_counter} step: query = {query}')
                else:
                    query = 0
                # print(f'lambd={self.lambd},c_dist={c_dist}, p_comm={p_comm}, rnd={rnd}, query={query}')
            
        self.pursuer_actions.append(pursuer_action)
        self.evader_actions.append(evader_action)

        # EVADER - distance between E and ~P before taking action
        dist_e_lp_before = self.compute_distance(self.evader, self.last_record_p_e)

        # PURSUER - distance between P and ~E before taking action
        dist_p_le_before = self.compute_distance(self.pursuer, self.last_record_e_p)

        # Update pursuer position
        u_i = pursuer_action # lateral acceleration
        u_i = np.clip(u_i, -self.pursuer_max_acceleration, self.pursuer_max_acceleration)
        self.pursuer['psi'] += u_i * self.dt  # / self.pursuer_speed
        self.pursuer['psi'] = self._normalize_angle(self.pursuer['psi'])
        self.pursuer['x'] += self.pursuer_speed * np.cos(self.pursuer['psi']) * self.dt
        self.pursuer['y'] += self.pursuer_speed * np.sin(self.pursuer['psi']) * self.dt
        # Detect collision with map boundary
        if (self.pursuer['x'] <= 0) | (self.pursuer['x'] >= self.map_size[0]) | (self.pursuer['y'] <= 0) | (
                self.pursuer['y'] >= self.map_size[1]):
            p_hit_boundary = True
        else:
            p_hit_boundary = False
        # Keep pursuer within bounds
        self.pursuer['x'] = np.clip(self.pursuer['x'], 0, self.map_size[0])
        self.pursuer['y'] = np.clip(self.pursuer['y'], 0, self.map_size[1])

        # Update evader
        u_e = evader_action
        u_e = np.clip(u_e, -self.evader_max_acceleration, self.evader_max_acceleration)
        self.evader['psi'] += u_e * self.dt  # / self.evader_speed
        self.evader['psi'] = self._normalize_angle(self.evader['psi'])
        self.evader['x'] += self.evader_speed * np.cos(self.evader['psi']) * self.dt
        self.evader['y'] += self.evader_speed * np.sin(self.evader['psi']) * self.dt
        # Detect collision with map boundary
        if (self.evader['x'] <= 0) | (self.evader['x'] >= self.map_size[0]) | (self.evader['y'] <= 0) | (
                self.evader['y'] >= self.map_size[1]):
            e_hit_boundary = True
        else:
            e_hit_boundary = False
        # Keep evader within bounds
        self.evader['x'] = np.clip(self.evader['x'], 0, self.map_size[0])
        self.evader['y'] = np.clip(self.evader['y'], 0, self.map_size[1])

        # Compute current state
        global_state = self._get_state()

        # Compute pursuer's observation
        # TODO: Consider returning here the difference from last observed state. But then zeros is like I queried
        # TODO: and got the answer that the evader didn't move.
        if query == 1:
            communication = True
            if self.noise_std == 0:
                pursuer_observation = self._get_pursuer_state(global_state)
                evader_observation = self._get_evader_state(global_state)
            else:
                o1, o2, o3, noise_x_p, noise_y_p = self._get_pursuer_state(global_state)
                pursuer_observation= (o1, o2, o3)
                o1, o2, o3, noise_x_e, noise_y_e = self._get_evader_state(global_state)
                evader_observation= (o1, o2, o3)
                # pursuer_observation, noise_x_p, noise_y_p = self._get_pursuer_state(global_state)
                # evader_observation, noise_x_e, noise_y_e = self._get_evader_state(global_state)
            
            # Updated lastly observed opponent position
            if self.noise_std == 0:
                self.last_record_e_p = self.evader.copy()
                self.last_record_p_e = self.pursuer.copy()
            else:
                self.last_record_e_p = {
                    'x': global_state[5]+ noise_x_p,
                    'y': global_state[6]+ noise_y_p,
                    'psi': global_state[7],
                }
                self.last_record_p_e = {
                    'x': global_state[0]+ noise_x_e,
                    'y': global_state[1]+ noise_y_e,
                    'psi': global_state[2],
                }
                
                

            # Nullify list of observed actions
            self.last_query_step = self.step_counter
            self.pursuer_actions = []
            self.evader_actions = []
        else:
            communication = False

            # distance between current position of pursuer and last recorded position of the evader
            dist = self.compute_distance(self.pursuer, self.last_record_e_p)
            pursuer_observation = np.array(
                [self.pursuer['x'], self.pursuer['y'], self.pursuer['psi'], self.pursuer_speed,
                 self.pursuer_max_acceleration] +
                [self.last_record_e_p['x'] - self.pursuer['x'], self.last_record_e_p['y'] - self.pursuer['y'],
                 self._normalize_angle(self.last_record_e_p['psi'] - self.pursuer['psi']),
                 self.evader_speed - self.pursuer_speed, self.evader_max_acceleration - self.pursuer_max_acceleration] +
                [dist, self.r_min, self.r_shoot, (self.step_counter - self.last_query_step)/self.max_steps,  self.last_query_step/self.max_steps], dtype=np.float64).flatten()
            pursuer_observation = (pursuer_observation, [], [])

            # distance between current position of evader and last recorded position of the pursuer
            dist = self.compute_distance(self.evader, self.last_record_p_e)
            evader_observation = np.array([self.evader['x'], self.evader['y'], self.evader['psi'], self.evader_speed,
                                           self.evader_max_acceleration] +
                                          [self.last_record_p_e['x'] - self.evader['x'],
                                           self.last_record_p_e['y'] - self.evader['y'],
                                           self._normalize_angle(self.last_record_p_e['psi'] - self.evader['psi']),
                                           self.pursuer_speed - self.evader_speed,
                                           self.pursuer_max_acceleration - self.evader_max_acceleration] +
                                          [dist, self.r_min, self.r_shoot, (self.step_counter - self.last_query_step)/self.max_steps,  self.last_query_step/self.max_steps], dtype=np.float64).flatten()
            evader_observation = (evader_observation, [], [])

        assert pursuer_observation[0].shape == self.observation_space[0].shape, \
            f"Expected shape {self.observation_space[0].shape}, got {pursuer_observation[0].shape}"
        assert evader_observation[0].shape == self.observation_space[0].shape, \
            f"Expected shape {self.observation_space[0].shape}, got {evader_observation[0].shape}"

        # Compute rewards
        if communication:
            delta_p = self.compute_distance(self.evader, self.pursuer) - dist_p_le_before
            delta_e = self.compute_distance(self.evader, self.pursuer) - dist_e_lp_before
        else:
            delta_p = self.compute_distance(self.pursuer, self.last_record_e_p) - dist_p_le_before
            delta_e = self.compute_distance(self.evader, self.last_record_p_e) - dist_e_lp_before

        # Check if game is over
        # p_win, p_loose = self.check_game_over(communication)

        p_win, p_loose, rew_p, rew_e = self.compute_rewards(p_lat_acc=u_i, e_lat_acc=u_e,
                                                            communication=communication,
                                                            delta_p=delta_p, delta_e=delta_e,
                                                            p_hit=p_hit_boundary, e_hit=e_hit_boundary)

        truncation = (self.step_counter >= self.max_steps) | (communication & p_loose)

        # record agents positions in history
        self.record_history(global_state)
        # Return observations, rewards, done flag, and info
        observations = {'pursuer_observation': pursuer_observation, 'evader_observation': evader_observation}
        rewards = {'rew_p': rew_p, 'rew_e': rew_e}
        done = p_win or truncation
        info = {'termination': p_win, 'truncation': truncation, 'state': global_state, 'actions': actions}
        return observations, rewards, done, info


    def record_history(self, state):
        self.history = self.history + [state]

    def empty_history(self):
        # self.history = [self.history[0]]
        self.history = []

    def _normalize_angle(self, angle):
        # Normalize angle to be within [-pi, pi]
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _get_state(self):
        # Return observations (positions and velocities of pursuers and evader)
        observations = [self.pursuer['x'], self.pursuer['y'], self.pursuer['psi'], self.pursuer_speed,
                        self.pursuer_max_acceleration] + \
                       [self.evader['x'], self.evader['y'], self.evader['psi'], self.evader_speed,
                        self.evader_max_acceleration] + \
                       [self.compute_distances(), self.r_min, self.r_shoot, 
                        (self.step_counter - self.last_query_step)/self.max_steps, self.last_query_step/self.max_steps]

        return np.array(observations).flatten()

    def _get_pursuer_state(self, global_state):
        state = copy.deepcopy(global_state)

        state[0] = global_state[0]  # x_p
        state[1] = global_state[1]  # y_p
        state[2] = global_state[2]  # psi_p
        state[3] = global_state[3]  # v_p
        state[4] = global_state[4]  # a_p

        if self.noise_std == 0:
            state[5] = global_state[5] - state[0]  # x_e-x_p
            state[6] = global_state[6] - state[1]  # y_e-y_p
        else:
            noise_x, noise_y = np.random.normal(0, self.noise_std, size=(2,))
            state[5] = global_state[5]+ noise_x - state[0]  # (x_e+noise_x)-x_p
            state[6] = global_state[6]+ noise_y - state[1]  # (y_e+noise_y)-y_p
            
        state[7] = self._normalize_angle(global_state[7] - state[2])  # psi_e-psi_p
        state[8] = global_state[8] - state[3]  # v_e-v_p
        state[9] = global_state[9] - state[4]  # a_e-a_p

        state[10] = global_state[10]  # dist
        state[11] = global_state[11]  # r_min
        state[12] = global_state[12]  # r_shoot
        if self.noise_std == 0:
            return state, self.evader_actions, self.pursuer_actions
        else:
            return state, self.evader_actions, self.pursuer_actions, noise_x, noise_y


    def _get_evader_state(self, global_state):
        state = copy.deepcopy(global_state)

        state[0] = global_state[5]  # x_e
        state[1] = global_state[6]  # y_e
        state[2] = global_state[7]  # psi_e
        state[3] = global_state[8]  # v_e
        state[4] = global_state[9]  # a_e

        if self.noise_std == 0:
            state[5] = global_state[0] - state[5]  # x_p-x_e
            state[6] = global_state[1] - state[6]  # y_p-y_e
        else:
            noise_x, noise_y = np.random.normal(0, self.noise_std, size=(2,))
            state[5] = global_state[0] + noise_x - state[5]  # x_p-x_e
            state[6] = global_state[1] + noise_y - state[6]  # y_p-y_e
            
        state[7] = self._normalize_angle(global_state[2] - state[7])  # psi_p-psi_e
        state[8] = global_state[3] - state[8]  # v_p-v_e
        state[9] = global_state[4] - state[9]  # a_p-a_e

        state[10] = global_state[10]  # dist
        state[11] = global_state[11]  # r_min
        state[12] = global_state[12]  # r_shoot
        
        if self.noise_std == 0:
            return state, self.pursuer_actions, self.evader_actions
        else:
            return state, self.pursuer_actions, self.evader_actions, noise_x, noise_y



    def compute_distances(self):
        abs_x_dist = np.abs(self.pursuer['x'] - self.evader['x'])
        abs_y_dist = np.abs(self.pursuer['y'] - self.evader['y'])

        distance_to_evader = np.sqrt(abs_x_dist ** 2 + abs_y_dist ** 2)

        return distance_to_evader

    def compute_rewards(self, p_lat_acc, e_lat_acc, delta_p=0, delta_e=0, communication=False, p_hit=False,
                        e_hit=False):

        p_win, p_loose = self.check_game_over(communication)

        if (self.step_counter >= self.max_steps):
            penalty = self.beta
        else:
            penalty = 0

        rew_p = self.beta * int(p_win) \
                - self.shooting_penalty * int(p_loose) \
                - penalty \
                - self.distance_penalty * (delta_p + 1e-7) \
                - self.communication_penalty * communication \
                - self.time_penalty \
                - self.hit_boundary_penalty * p_hit \
                - self.lat_acceleration_penalty * np.abs(p_lat_acc)

        rew_e = - self.beta * int(p_win) \
                + penalty \
                + self.distance_penalty * (delta_e + 1e-7) \
                + self.time_penalty \
                - self.hit_boundary_penalty * e_hit \
                - self.lat_acceleration_penalty * np.abs(e_lat_acc)

        return p_win, p_loose, rew_p, rew_e

    def exponential_cdf(self, distance):
        return np.exp2(-self.lambd * distance / self.r_shoot)

    def check_game_over(self, communication):
        # Termination condition

        p_win = False
        p_loose = True
        if self.probabilistic:
            p_shoot = self.exponential_cdf(self.compute_distances())
            if (random.random() < p_shoot) & communication:
                return p_win, p_loose
            else:
                p_loose = False
        else:
            if (bool(np.min(self.compute_distances()) < self.r_shoot)) & (communication):
                return p_win, p_loose
            else:
                p_loose = False

        if bool(np.min(self.compute_distances()) <= self.r_min):
            p_win = True
            return p_win, p_loose

        return p_win, p_loose
