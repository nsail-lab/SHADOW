from functools import partial

import numpy as np
import matplotlib#; matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from copy import deepcopy
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

class DifferentialGameEnvironment(gym.Env):

    """Configurable environment for maze. """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, num_pursuers,
                 map_size,
                 pursuer_speed,
                 evader_speed,
                 pursuer_max_acceleration,
                 evader_max_acceleration,
                 r_min,
                 dt=0.01,
                 max_steps=10_000,
                 beta=0.05):
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
        self.beta = beta
        self.pursuer = []
        self.evader = []
        self.history = []

        self.evader_actions = []

        # Action space: one numeric value for lateral acceleration for each agent
        self.action_space = spaces.Box(low=-self.pursuer_max_acceleration, high=self.pursuer_max_acceleration, shape=(1,), dtype=np.float64)

        # Observation space: positions (x, y) and heading angles (psi) for both the pursuer and the evader
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi, 0.0, 0.0, -np.pi, 0]),
            high=np.array([1.0, 1.0, np.pi, 1.0, 1.0, np.pi, np.inf]),
            dtype=np.float64
        )

        self.reset()
        #self.record_history(self._get_observations())

    def reset(self, seed = None):
        if seed:
            np.random.seed(seed=seed)

        # Initialize step counter
        self.step_counter = 0

        self.evader_actions = []
        
        self.pursuer = {'x': np.random.uniform(0, self.map_size[0]),  # x coordinate
                          'y': np.random.uniform(0, self.map_size[1]),  # y coordinate
                          'psi': np.random.uniform(-np.pi, np.pi),  # heading angle
                          't': 0}  # when to query the state next
        

        self.evader = {'x': np.random.uniform(0, self.map_size[0]),  # x coordinate
                       'y': np.random.uniform(0, self.map_size[1]),  # y coordinate
                       'psi': np.random.uniform(-np.pi,  np.pi)}  # heading angle
        
        self.empty_history()
        self.record_history(self._get_observations())

        state = self.history[0]
        self.evader['full_state'] = deepcopy(state)
        self.pursuer['full_state'] = deepcopy(state)
        #for i in range(self.num_pursuers):
        #    self.pursuers[i]['full_state'] = deepcopy(state)

        return self._get_observations(), {}
    
    def render(self, mode='human', close=False):
        
        x_p, y_p, psi_p, x_e, y_e, psi_e, dist = self._get_observations()
        if mode == 'rgb_array':
            fig, ax = plt.subplots(1,1,figsize=(4,3),dpi=150)
            ax.scatter(x_p, y_p,s=30, color='blue')
            ax.scatter(x_e, y_e,s=30, color='red',marker='*')
            ax.grid()
            plt.show()
        else:
            return self._get_observations()



    def step(self, pursuer_actions):
        self.step_counter += 1

        # Update pursuers
        
        u_i = pursuer_actions[0]  # lateral acceleration
        u_i = np.clip(u_i, -self.pursuer_max_acceleration, self.pursuer_max_acceleration)
        self.pursuer['psi'] += u_i * self.dt  # / self.pursuer_speed
        self.pursuer['psi'] = self._normalize_angle(self.pursuer['psi'])

        self.pursuer['x'] += self.pursuer_speed * np.cos(self.pursuer['psi']) * self.dt
        self.pursuer['y'] += self.pursuer_speed * np.sin(self.pursuer['psi']) * self.dt

        # Keep pursuers within bounds
        self.pursuer['x'] = np.clip(self.pursuer['x'], 0, self.map_size[0])
        self.pursuer['y'] = np.clip(self.pursuer['y'], 0, self.map_size[1])
        self.pursuer['t'] -= 1

        # Update evader
        u_e = self.action_space.sample()[0] #evader_action[0]

        self.evader_actions.append(u_e)

        u_e = np.clip(u_e, -self.evader_max_acceleration, self.evader_max_acceleration)
        self.evader['psi'] += u_e * self.dt  # / self.evader_speed
        self.evader['psi'] = self._normalize_angle(self.evader['psi'])

        self.evader['x'] += self.evader_speed * np.cos(self.evader['psi']) * self.dt
        self.evader['y'] += self.evader_speed * np.sin(self.evader['psi']) * self.dt
        # Keep evader within bounds
        self.evader['x'] = np.clip(self.evader['x'], 0, self.map_size[0])
        self.evader['y'] = np.clip(self.evader['y'], 0, self.map_size[1])

        #print(f"[valerio] - evader(psi, x,y) = {self.evader['psi']} | {self.evader['x']} | {self.evader['y']}")

        # Compute current state
        state = self._get_observations()


        #self.pursuer['t'] = pursuer_actions[0][1]  # update when to query full state next
        self.pursuer['full_state'] = deepcopy(state)  # give pursuer the current full state
        self.evader['full_state'] = deepcopy(state)


        # Compute rewards
        rewards = self.compute_rewards()

        # Check if game is over
        termination = self.check_game_over() 
        truncation = self.step_counter >= self.max_steps

        # record agents positions in history
        self.record_history(state)
        # Return observations, rewards, and done flag
        return self._get_observations(), rewards, termination, truncation,{}

    def record_history(self, state):
        self.history = self.history + [state]

    def empty_history(self):
        # self.history = [self.history[0]]
        self.history = []

    def _normalize_angle(self, angle):
        # Normalize angle to be within [-pi, pi]
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _get_observations(self):
        # Return observations (positions and velocities of pursuers and evader)
        #observations = []
        #for pursuer in self.pursuers:
        #    observations.append([self.pursuer['x'], self.pursuer['y'], self.pursuer['psi']])
        #
        #observations = [item for items in observations for item in items]
        #observations.append([self.evader['x'], self.evader['y'], self.evader['psi']])
        
        observations = [self.pursuer['x'], self.pursuer['y'], self.pursuer['psi']] + \
                       [self.evader['x']-self.pursuer['x'], self.evader['y']-self.pursuer['y'], self._normalize_angle(self.evader['psi']-self.pursuer['psi'])] + \
                       [self.compute_distances()]
        
        # observations = [self.pursuer['x'], self.pursuer['y'], self.pursuer['psi']] + \
        #                [self.evader['x'], self.evader['y'], self.evader['psi']] + \
        #                [self.compute_distances()]
        return np.array(observations)

    def _compute_state(self, state, i):
        updatable_pursuers = [j for j in range(self.num_pursuers) if self.pursuers[j]['t'] > 0 or j == i]
        for j in updatable_pursuers:
            self.pursuers[i]['full_state'][j] = deepcopy(state[j])

    def _parse_state(self, state, i):
        # for evader:
        if i == -1:
            # evader, then the rest
            return np.array([state[-1], *state[:-1]])

        # agent i's state, then evader, then the rest
        return np.array([state[i], state[-1], *state[:i], *state[i + 1:-1]])

    def _parse_observations(self):
        return np.array([self._parse_state(self.pursuers[i]['full_state'], i) for i in range(self.num_pursuers)]), self._parse_state(self.evader['full_state'], -1)

    def compute_distances(self):

        
        abs_x_dist = np.abs(self.pursuer['x'] - self.evader['x'])
        abs_y_dist = np.abs(self.pursuer['y'] - self.evader['y'])
        
        distance_to_evader = np.sqrt(abs_x_dist**2 + abs_y_dist**2)

        return distance_to_evader

    def compute_rewards(self):
        # Placeholder reward function
        # For example, negative distance to evader for pursuers and positive distance for evader
        
        distances = -self.compute_distances()
        done = self.check_game_over()
        rewards = 1000 * self.beta * (done) + self.beta * distances #- self.dt * 0.1
        #rewards = self.beta * (not done) + self.beta * distances
        #rewards /= self.max_steps  # scale rewards to stabilize critic training
        return rewards

    def check_game_over(self):
        # Placeholder termination condition
        return bool(np.min(self.compute_distances()) <= self.r_min)

    # Update function for animation
    def update(self, frame, pursuers_objects, evader_object, arrow_length):
        for i in range(self.num_pursuers):
            pursuer_x = self.history[frame][i][0]
            pursuer_y = self.history[frame][i][1]
            pursuer_psi = self.history[frame][i][2]
            pursuers_objects[i].set_data(x=pursuer_x, y=pursuer_y,
                                         dx=arrow_length*np.cos(pursuer_psi),
                                         dy=arrow_length*np.sin(pursuer_psi))

        evader_x = self.history[frame][-1][0]
        evader_y = self.history[frame][-1][1]
        evader_psi = self.history[frame][-1][2]
        evader_object.set_data(x=evader_x, y=evader_y,
                               dx=arrow_length * np.cos(evader_psi),
                               dy=arrow_length * np.sin(evader_psi))

        return *pursuers_objects, evader_object

    def history_animation(self, episode, arrow_length=0.05, arrow_width=0.01):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.map_size[0])
        ax.set_ylim(0, self.map_size[1])
        plt.title(f'Episode {episode}')       

        evader_states = [s[1].tolist() for s in self.history]
        pursuer_states = [s[0].tolist() for s in self.history]
        df_e_positions = pd.DataFrame(evader_states,columns=['x','y','psi'])
        df_p_positions = pd.DataFrame(pursuer_states,columns=['x','y','psi'])

        #ax.plot(df_e_positions.x,df_e_positions.y,lw=2,label='evader')
        ax.plot(df_p_positions.x,df_p_positions.y,lw=2,label='pursuer')
        ax.scatter(df_e_positions.x.iloc[0],df_e_positions.y.iloc[0],color='red',s=30)
        plt.show()

    def history_animation_OLD(self, episode, arrow_length=0.05, arrow_width=0.01):
        # Create a figure and axis
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.map_size[0])
        ax.set_ylim(0, self.map_size[1])
        plt.title(f'Episode {episode}')

        # Create pursuer objects
        pursuers_objects = []
        for i in range(self.num_pursuers):
            pursuer_x = self.history[0][i][0]
            pursuer_y = self.history[0][i][1]
            pursuer_psi = self.history[0][i][2]
            pursuers_objects.append(patches.FancyArrow(x=pursuer_x,
                                                       y=pursuer_y,
                                                       dx=arrow_length*np.cos(pursuer_psi),
                                                       dy=arrow_length*np.sin(pursuer_psi),
                                                       width=arrow_width,
                                                       color='blue'))
            ax.add_patch(pursuers_objects[i])  # Add the blue pursuer arrow to the plot

        # Create evader object
        evader_x = self.history[0][-1][0]
        evader_y = self.history[0][-1][1]
        evader_psi = self.history[0][-1][2]
        evader_object = patches.FancyArrow(x=evader_x,
                                           y=evader_y,
                                           dx=arrow_length*np.cos(evader_psi),
                                           dy=arrow_length*np.sin(evader_psi),
                                           width=arrow_width,
                                           color='red')
        ax.add_patch(evader_object)  # Add the red evader arrow to the plot

        # Create animation
        ani = FuncAnimation(fig, partial(self.update,
                                         pursuers_objects=pursuers_objects,
                                         evader_object=evader_object,
                                         arrow_length=arrow_length),
                            frames=len(self.history),
                            interval=1,
                            repeat=True)
        plt.show()

if __name__ == '__main__':
    import os
    import sys
    import pandas as pd

    sys.path.append('./config')
    sys.path.append('./model')
    from config import Config
    from TD3 import TD3Agent
    import copy 

    args = Config()
    
    # Initialize environment
    env_kwargs = {
        "num_pursuers": args.num_pursuers,
        "map_size":  (args.map_width, args.map_height),
        "pursuer_speed": args.pursuer_speed,
        "evader_speed": args.evader_speed,
        "pursuer_max_acceleration": args.pursuer_max_acceleration,
        "evader_max_acceleration": args.evader_max_acceleration,
        "r_min": args.r_min,
        "dt": args.dt,
        "beta": args.beta,
        "max_steps": args.max_steps
    }
    env = DifferentialGameEnvironment(**env_kwargs)

    env.reset()

    agent_state_dim = 3  # x,y,psi coordinate system TODO: maybe also t
    state_dim = (env.num_pursuers + 1) * agent_state_dim  # state is composed of the state of each agent
    action_dim = 1

    model_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
        "discount": args.discount,
        "tau": args.tau,
        "expl_noise": args.expl_noise,
        "replay_buffer_size": args.replay_buffer_size,
        "batch_size": args.batch_size
    }

    # Initialize a single TD3 agent for all pursuers
    pursuer_model_kwargs = copy.deepcopy(model_kwargs)
    pursuer_model_kwargs["max_action"] = args.pursuer_max_acceleration
    pursuer_agent = TD3Agent(**pursuer_model_kwargs)
    # TODO: Action dim of pursuer is 2: a continuous u_i for lateral acceleration, and a binary control variable to
    #  enable querying information.

    # Initialize TD3 agent for evader
    evader_model_kwargs = copy.deepcopy(model_kwargs)
    evader_model_kwargs["max_action"] = args.evader_max_acceleration
    evader_agent = TD3Agent(**evader_model_kwargs)

    for i in range(1000):
        pursuer_actions = [pursuer_agent.sample_action() for _ in range(env.num_pursuers)]
        evader_action = evader_agent.sample_action()

        #print(f'{i} pursuer_actions = ', pursuer_actions)
        #print(f'{i} evader_action = ', evader_action)
        next_pursuer_states, next_evader_state, rewards, done = env.step(pursuer_actions, evader_action)

    evader_states = [s[1].tolist() for s in env.history]
    pursuer_states = [s[0].tolist() for s in env.history]



