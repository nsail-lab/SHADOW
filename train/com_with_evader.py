import copy
from argparse import ArgumentParser

import gymnasium as gym
from tqdm import tqdm
import numpy as np
import os

from mediators.Mediator import Mediator
from utils.evaluation import eval_policy, aggregate_results, compute_episode_statistics

def train(pursuer_agent, evader_agent,
          pursuer_mediator: Mediator, evader_mediator: Mediator,
          env, args):
    episode_timestep = 0
    episode_num = 0
    episode_rew_p = 0
    episode_rew_e = 0
    evaluations = []

    global_state, _ = env.reset()

    pursuer_state = pursuer_mediator.process_observation(env._get_pursuer_state(global_state))
    evader_state = evader_mediator.process_observation(env._get_evader_state(global_state))

    for t in range(args.max_timesteps):
        # Evaluate episode
        if episode_num % args.eval_freq == 0 and t > args.start_time_steps:
            avg_reward, meta = eval_policy(pursuer_agent, evader_agent,
                                           pursuer_mediator, evader_mediator,
                                           env,
                                           eval_episodes=args.eval_episodes,
                                           max_steps=args.max_steps,
                                           eval=True,
                                           verbose=False)
            df_raw_episodes = aggregate_results(meta)
            df_stats = df_raw_episodes.groupby('episode').apply(lambda episode_metadata: compute_episode_statistics(episode_metadata, args))

            evaluations.append(avg_reward)
            env.reset()
            episode_num += 1
            if (args.save_model):
                save_dir = os.path.join(args.save_model_path,str(episode_num))
                os.mkdir(save_dir)
                evader_dir = os.path.join(save_dir, 'evader')
                pursuer_dir = os.path.join(save_dir, 'pursuer')
                os.mkdir(evader_dir)
                os.mkdir(pursuer_dir)
                df_stats.to_csv(os.path.join(save_dir,'df_stats.csv'))

                pursuer_agent.save(pursuer_dir)
                evader_agent.save(evader_dir)

        episode_timestep += 1
        if t < args.start_time_steps:
            # Select actions for pursuers
            mov_action, com_action = pursuer_agent.sample_action(pursuer_state)
            mov_action_e = evader_agent.sample_action()
        else:
            # Select actions for pursuers
            mov_action, com_action = pursuer_agent.select_train_action(pursuer_state)
            mov_action_e = evader_agent.select_train_action(evader_state)

        action, probs, vals = com_action
        
        # Execute actions and observe next states and rewards

        actions = [np.array([mov_action[0], mov_action_e[0]]), action]  # TODO(Dolev): not natural, separate evader and pursuer actions

        observations, rewards, done, info = env.step(actions)

        next_pursuer_state = pursuer_mediator.process_observation(observations['pursuer_observation'])
        next_evader_state = evader_mediator.process_observation(observations['evader_observation'])

        # Store experiences in replay buffer for the pursuer
        # TODO(Dolev): we add the same reward to both pursuer networks.
        # TODO(Dolev): makes no sense to add com. penalty to TD3 move. net, or collision penalty to PPO com.
        pursuer_agent.store_buffer(mov_action, next_pursuer_state,
                                   action, probs, vals,
                                   pursuer_state, rewards['rew_p'], done)
        
        evader_agent.replay_buffer.add(evader_state,
                                       mov_action_e,
                                       rewards['rew_e'],
                                       next_evader_state,
                                       done)  # TODO(Dolev): args for e and p not same order...

        pursuer_state = next_pursuer_state.copy()
        evader_state = next_evader_state.copy()

        episode_rew_p += np.min(rewards['rew_p'])  # TODO(Dolev): why min?
        episode_rew_e += np.min(rewards['rew_e'])  # TODO(Dolev): why min?

        if t >= args.start_time_steps:
            # Update pursuer and evader agents
            pursuer_agent.train()
            evader_agent.train()

        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num + 1}, Episode T: {episode_timestep} Rew_p: {episode_rew_p}, Rew_e: {episode_rew_e}, Explor. Noise: {pursuer_agent.movement_agent.expl_noise}")

            global_state, _ = env.reset()

            pursuer_state = pursuer_mediator.process_observation(env._get_pursuer_state(global_state))
            evader_state = evader_mediator.process_observation(env._get_evader_state(global_state))

            pursuer_agent.clear_history()
            evader_agent.clear_history()

            if t > args.start_time_steps:
                pursuer_agent.movement_agent.expl_noise *= args.expl_noise_decay
                pursuer_agent.movement_agent.expl_noise = max(pursuer_agent.movement_agent.expl_noise, args.min_expl_noise)

                evader_agent.expl_noise *= args.expl_noise_decay
                evader_agent.expl_noise = max(evader_agent.expl_noise, args.min_expl_noise)     

            # Update counters
            episode_rew_p = 0
            episode_rew_e = 0
            episode_timestep = 0
            episode_num += 1

            