import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import numpy as np
import scipy.stats
from time import time

from mediators.Mediator import Mediator


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def compute_episode_statistics(episode, args):
    df_states = episode.copy()
    termination = df_states.termination.sum()>0
    truncation = len(df_states) == args.max_steps
    shooting = 0 < df_states.truncation.sum() < args.max_steps
    steering_cost_p = df_states.p_movement.apply(lambda x: np.abs(x)).sum()
    steering_cost_e = df_states.e_movement.apply(lambda x: np.abs(x)).sum()
    collision_boundary_p = ((df_states['x_p'] == 0) | (df_states['x_p'] == 1) | 
                            (df_states['y_p'] == 0) | (df_states['y_p'] == 1)).sum()
    collision_boundary_e = ((df_states['x_e'] == 0) | (df_states['x_e'] == 1) | 
                            (df_states['y_e'] == 0) | (df_states['y_e'] == 1)).sum()
    comm_min_distance = df_states[df_states.p_communication==1].dist.min()

    # determine avg and median distance between communications
    indices = df_states.index[df_states['p_communication'] == 1].to_numpy()
    if len(indices) > 1:
        gaps = np.diff(indices)
        avg_gap = np.mean(gaps)
        median_gap = np.median(gaps)
    else:
        avg_gap = None  # Not enough occurrences of 1 to compute
        median_gap = None

    rew_p = df_states.rews_p.sum()
    rew_e = df_states.rews_e.sum()
    length = len(df_states)
    # print(df_states[['p_communication','p_movement']].head())
    # print(df_states.p_communication.head(10))
    comm_ratio = df_states.p_communication.sum()/length

    return pd.Series({
        'termination': termination,
        'truncation': truncation,
        'shooting': shooting,
        'steering_cost_p': steering_cost_p,
        'steering_cost_e': steering_cost_e,
        'collision_boundary_p': collision_boundary_p,
        'collision_boundary_e': collision_boundary_e,
        'comm_min_distance': comm_min_distance,
        'rew_p': rew_p,
        'rew_e': rew_e,
        'length': length,
        'comm_ratio': comm_ratio,
        'comm_gap_avg': avg_gap,
        'comm_gap_median': median_gap
    })


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(pursuer_agent, evader_agent,
                pursuer_mediator: Mediator, evader_mediator: Mediator,
                env, eval_episodes=10, max_steps=1_000, 
                eval =False, verbose=False, 
                opponent_modelling_p=True, opponent_modelling_e=True, 
                is_pdqn=False, HyAR=False, MPDQN=False,
                is_multihead_e=False):
    avg_reward_p = 0.
    avg_reward_e = 0.
    meta = {}
    for seed in tqdm(range(eval_episodes), total=eval_episodes):
        global_state, _ = env.reset(seed=seed)
        # print(env.pursuer_speed/env.evader_speed)
        pursuer_state = env._get_pursuer_state(global_state)
        evader_state = env._get_evader_state(global_state)

        pursuer_mediator.reset(env)
        evader_mediator.reset(env)

        pursuer_state = pursuer_mediator.process_observation(pursuer_state)
        evader_state = evader_mediator.process_observation(evader_state)

        pursuer_agent.clear_history()
        evader_agent.clear_history()

        steps = 0
        done = False

        meta[seed] = {'states': [], 'rews_p':[], 'rews_e':[],
                      "p_states": [], "e_states": [],
                      'infos':[], 'n_steps':0, 'truncation':False, 'termination':False}
        meta[seed]['p_states'].append(pursuer_state)
        meta[seed]['e_states'].append(evader_state)
        meta[seed]['states'].append(global_state)

        rew_p = 0.
        rew_e = 0.
        while not done and steps < max_steps:
            steps += 1

            # start = time()
            if opponent_modelling_p:
                mov_action, com_action, opponent_action, uncertainty = pursuer_agent.select_action(pursuer_state)
            else:
                mov_action, com_action = pursuer_agent.select_action(pursuer_state)
            
            if opponent_modelling_e:
                # print(f"[DEBUG] select_action() took {time() - start:.6f} seconds")
                mov_action_e,opponent_action_e, uncertainty_e = evader_agent.select_action(evader_state)
            else:
                mov_action_e = evader_agent.select_action(evader_state)

            # Process pursuer actions
            if opponent_modelling_p:
                pursuer_mediator.process_action((mov_action[0], com_action, opponent_action, uncertainty))
            if opponent_modelling_e:
                evader_mediator.process_action((mov_action_e[0], com_action, opponent_action_e, uncertainty_e))
            
            if is_multihead_e:
                if is_pdqn:
                    actions = [np.array([mov_action, mov_action_e[0]]), com_action]  
                else:
                    actions = [np.array([mov_action[0], mov_action_e[0]]), com_action]

            else:
                if (is_pdqn==False) & (HyAR==False) & (MPDQN==False):
                    actions = [np.array([mov_action[0], mov_action_e[0]]), com_action]
                elif HyAR == True:
                    actions = [np.array([mov_action, mov_action_e[0]]), com_action]
                elif MPDQN == True:
                    actions = [np.array([mov_action, mov_action_e[0]]), com_action]
                else:
                    if opponent_modelling_p:
                        actions = [np.array([mov_action[0], mov_action_e[0]]), com_action]
                    else:
                        actions = [np.array([mov_action, mov_action_e[0]]), com_action]

            # start = time()
            observations, rewards, done, info = env.step(actions)
            # print(f"[DEBUG] env.step() took {time() - start:.6f} seconds")

            termination = info['termination']
            truncation = info['truncation']

            # start = time()
            pursuer_state = pursuer_mediator.process_observation(copy.deepcopy(observations['pursuer_observation']))
            # print(f"[DEBUG] Mediator processing took {time() - start:.6f} seconds")
            evader_state = evader_mediator.process_observation(copy.deepcopy(observations['evader_observation']))

            # start = time()
            meta[seed]['states'].append(info['state'])
            # print(f"[DEBUG] meta logging took {time() - start:.6f} seconds")
            meta[seed]['p_states'].append(pursuer_state)
            meta[seed]['e_states'].append(evader_state)
            meta[seed]['rews_p'].append(rewards['rew_p'])
            meta[seed]['rews_e'].append(rewards['rew_e'])
            meta[seed]['infos'].append(info)
            
            
            done = termination or truncation

            rew_p += rewards['rew_p']
            rew_e += rewards['rew_e']
        
        avg_reward_p += rew_p
        avg_reward_e += rew_e

        meta[seed]['n_steps'] = steps
        meta[seed]['termination'] = termination
        meta[seed]['truncation'] = truncation
        meta[seed]['lambd'] = env.lambd
        
        # print(np.unique(np.array(meta[seed]['states'])[:, 3], return_counts=True), np.unique(np.array(meta[seed]['states'])[:, 8], return_counts=True))
        
        if verbose:
            print(f'[DEBUG] - episode {seed} | rew_p {rew_p} | rew_e {rew_e}| n_steps {steps} ')

        pursuer_agent.clear_history()
        evader_agent.clear_history()
        
        #env.history_animation(total_reward, arrow_length=0.05, arrow_width=0.01)
    avg_reward_p /= eval_episodes
    avg_reward_e /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward_p:.3f}")
    print("---------------------------------------")

    if eval:
        return avg_reward_p, meta
    else:
        return avg_reward_p
    

def aggregate_results(meta):
    dfs = []

    for i, data in meta.items():
        # Convert the dictionary into a DataFrame
        df = pd.DataFrame({
            "states": [list(state) for state in data["states"]],  # Convert arrays to lists
            "p_states": [list(p_state) for p_state in data["p_states"]],  # Convert arrays to lists
            "n_steps": data["n_steps"],
            "truncation": data["truncation"],
            "termination": data["termination"],
            "lambd": data["lambd"],
            # "infos": [None] + [list(item['env_state']) for item in data["infos"]],
            "rews_p": [None] + [reward for reward in data["rews_p"]],
            "rews_e": [None] + [reward for reward in data["rews_e"]],
            "actions": [None] + [info['actions'] for info in data["infos"]],
        })
        df = df.iloc[1:]
        df.index = df.reset_index().index
        
        df['episode'] = i

        df['p_movement'] = df.actions.apply(lambda x: x[0][0] if x is not None else None)
        df['p_communication'] = df.actions.apply(lambda x: x[1] if x is not None else None)
        df['e_movement'] = df.actions.apply(lambda x: x[0][1] if x is not None else None)
        df = df.drop('actions',axis=1)

        # Expand the 'states' column into multiple columns
        if len(df["states"].to_list()[0])==7:
            states_expanded = pd.DataFrame(df["states"].to_list(), columns=['x_p','y_p','psi_p','x_e','y_e','psi_e','dist'])
        elif len(df["states"].to_list()[0])==15:
            states_expanded = pd.DataFrame(df["states"].to_list(), columns=['x_p','y_p','psi_p','v_p','a_p','x_e','y_e','psi_e','v_e','a_e','dist','r_min','r_shoot','T_since_last_comm','T_last_comm'])
        else:
            states_expanded = pd.DataFrame(df["states"].to_list(), columns=['x_p','y_p','psi_p','v_p','a_p','x_e','y_e','psi_e','v_e','a_e','dist','r_min','r_shoot',])

        # Combine the expanded DataFrame with the original, dropping the old 'states' column
        df = pd.concat([df.drop(columns=['states']), states_expanded], axis=1)

        # Expand the 'p_states' column into multiple columns
        if (len(df["p_states"].to_list()[0])==17) | (len(df["p_states"].to_list()[0])==19):
            # Pull out evader state prediction and uncertainty columns
            if len(df["p_states"].to_list()[0])==19:
                p_states_expanded = pd.DataFrame(df["p_states"].to_list(),
                                                columns=['x_p','y_p','psi_p','v_p','a_p',
                                                        'dx_e0','dy_e0','dpsi_e0','dv_e','da_e','dist','r_min','r_shoot','T_since_last_comm','T_last_comm',
                                                        'dx_e_pred', 'dy_e_pred', 'dpsi_e_pred', 'sigma'])
            else:
                p_states_expanded = pd.DataFrame(df["p_states"].to_list(),
                                                columns=['x_p','y_p','psi_p','v_p','a_p',
                                                        'dx_e0','dy_e0','dpsi_e0','dv_e','da_e','dist','r_min','r_shoot',
                                                        'dx_e_pred', 'dy_e_pred', 'dpsi_e_pred', 'sigma'])
            
            # Compute evader estimated state from diffs
            p_states_expanded[['x_e_pred', 'y_e_pred', 'psi_e_pred']] = (p_states_expanded[['x_p','y_p','psi_p']].to_numpy() +
                                                                         p_states_expanded[['dx_e0','dy_e0','dpsi_e0']].to_numpy() +
                                                                         p_states_expanded[['dx_e_pred', 'dy_e_pred', 'dpsi_e_pred']].to_numpy())
            # Normalize angle
            p_states_expanded['psi_e_pred'] = (p_states_expanded['psi_e_pred'] + np.pi) % (2 * np.pi) - np.pi

            # Combine the expanded DataFrame with the original, dropping the old 'p_states' column
            df = pd.concat([df.drop(columns=['p_states']),
                            p_states_expanded[['x_e_pred', 'y_e_pred', 'psi_e_pred','sigma']]],
                           axis=1)

        # df_results = pd.concat([df_results,df])
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)