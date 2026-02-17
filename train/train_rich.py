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

    pursuer_state = env._get_pursuer_state(global_state)
    # print('[Valerio] DEBUG - pursuer_state = ', pursuer_state)
    
    evader_state = env._get_evader_state(global_state)

    pursuer_mediator.reset(env)
    evader_mediator.reset(env)

    pursuer_state = pursuer_mediator.process_observation(pursuer_state)
    evader_state = evader_mediator.process_observation(evader_state)

    # print('[Valerio] DEBUG - pursuer_state.shape = ', pursuer_state.shape)
    
    for t in range(args.max_timesteps):
        # Evaluate episode
        if episode_num % args.eval_freq == 0 and t > args.start_time_steps:
            avg_reward, meta = eval_policy(pursuer_agent, evader_agent,
                                           pursuer_mediator, evader_mediator,
                                           env,
                                           eval_episodes=args.eval_episodes,
                                           max_steps=args.max_steps,
                                           opponent_modelling_e=args.opponent_modelling_e,
                                           opponent_modelling_p=args.opponent_modelling_p,
                                           eval=True,
                                           verbose=False,
                                           is_pdqn=args.PDQN,
                                           HyAR=args.HyAR,
                                           MPDQN=args.MPDQN,
                                           is_multihead_e=args.Evader_MHPPO)
            df_raw_episodes = aggregate_results(meta)
            df_stats = df_raw_episodes.groupby('episode').apply(
                lambda episode_metadata: compute_episode_statistics(episode_metadata, args))

            evaluations.append(avg_reward)

            global_state, _ = env.reset()

            pursuer_state = env._get_pursuer_state(global_state)
            evader_state = env._get_evader_state(global_state)
            
            
            pursuer_mediator.reset(env)
            evader_mediator.reset(env)

            pursuer_state = pursuer_mediator.process_observation(pursuer_state)
            evader_state = evader_mediator.process_observation(evader_state)
            

            pursuer_agent.clear_history()
            evader_agent.clear_history()

            episode_num += 1
            if (args.save_model):
                
                save_dir = os.path.join(args.save_model_path, str(episode_num))
                os.mkdir(save_dir)
                evader_dir = os.path.join(save_dir, 'evader')
                pursuer_dir = os.path.join(save_dir, 'pursuer')
                os.mkdir(evader_dir)
                os.mkdir(pursuer_dir)
                df_stats.to_csv(os.path.join(save_dir, 'df_stats.csv'))

                pursuer_agent.save(pursuer_dir)
                evader_agent.save(evader_dir)

        episode_timestep += 1
        if t < args.start_time_steps:
            # Select actions for pursuers
            
            if args.opponent_modelling_p:
                mov_action, com_action, opponent_action, uncertainty = pursuer_agent.sample_action(pursuer_state)
            else:
                mov_action, com_action = pursuer_agent.sample_action(pursuer_state)

            
            if args.opponent_modelling_e:
                    
                mov_action_e, opponent_action_e, uncertainty_e = evader_agent.sample_action(evader_state)
            else:
                if args.Evader_MHPPO:
                    mov_action_e, logprob, value = evader_agent.sample_action(evader_state)
                elif args.Evader_PDQN:
                    mov_action_e, _ = evader_agent.sample_action(evader_state)
                else:
                    mov_action_e = evader_agent.sample_action()
        else:
            # Select actions for pursuers
            if args.opponent_modelling_p:
                mov_action, com_action, opponent_action, uncertainty = pursuer_agent.select_train_action(pursuer_state)
            else:
                mov_action, com_action = pursuer_agent.select_train_action(pursuer_state)
                
            if args.opponent_modelling_e:
                mov_action_e, opponent_action_e, uncertainty_e = evader_agent.select_train_action(evader_state)
            else:
                if args.Evader_MHPPO:
                    mov_action_e, logprob, value = evader_agent.select_action(evader_state)
                elif args.Evader_PDQN:
                    mov_action_e, _ = evader_agent.select_train_action(evader_state)
                else:
                    mov_action_e = evader_agent.select_train_action(evader_state)
        

        
        # Process pursuer actions
        if args.opponent_modelling_p:
            pursuer_mediator.process_action((mov_action[0], com_action[0], opponent_action, uncertainty))
        else:
            pass
        if args.opponent_modelling_e:
            evader_mediator.process_action((mov_action_e[0], com_action[0], opponent_action_e, uncertainty_e))
            # evader_mediator.process_action((mov_action_e[0], com_action, opponent_action_e, uncertainty_e))
        else:
            pass

        # Unpack communication action
        if args.Evader_MHPPO:

            if args.PDQN:
                
                actions = [np.array([mov_action, mov_action_e]), com_action]  
                action = com_action
                probs = 0
                vals = 0
                
            else:
                action, probs, vals = com_action
                actions = [np.array([mov_action[0], mov_action_e]), com_action] 
                 
       
        else:
            if (not args.PDQN) & (not args.HyAR) & (not args.MPDQN) & (not args.Evader_PDQN):
                action, probs, vals = com_action
                
                # Execute actions and observe next states and rewards
                # TODO(Dolev): not natural, separate evader and pursuer actions
                actions = [np.array([mov_action[0], mov_action_e[0]]), action]         

            elif args.HyAR:
                actions = [np.array([mov_action, mov_action_e[0]]), com_action]
                action = com_action
                probs = 0
                vals = 0
            elif args.MPDQN:
                actions = [np.array([mov_action, mov_action_e[0]]), com_action]
                action = com_action
                probs = 0
                vals = 0                  
            else:                
                if args.opponent_modelling_p:
                    if args.Evader_PDQN:
                        action, probs, vals = com_action
                        actions = [np.array([mov_action[0], mov_action_e]), action]
                    else:
                        action, probs, vals = com_action
                        actions = [np.array([mov_action[0], mov_action_e[0]]), action]
                else:
                    
                    # if args.Evader_PDQN:
                    #     if args.PDQN:
                    #         actions = [np.array([mov_action, mov_action_e]), com_action]
                    #         action = com_action
                    #         probs = 0
                    #         vals = 0
                    #     else:
                    #         actions = [np.array([mov_action[0], mov_action_e]), com_action]
                    #         action = com_action
                    #         probs = 0
                    #         vals = 0
                    # else:
                    actions = [np.array([mov_action, mov_action_e[0]]), com_action]
                    action = com_action
                    probs = 0
                    vals = 0
                
        
        observations, rewards, done, info = env.step(actions)
        
        next_pursuer_state = pursuer_mediator.process_observation(observations['pursuer_observation'],
                                                                  info=info)
        next_evader_state = evader_mediator.process_observation(observations['evader_observation'],
                                                                info=info)

        # Store experiences in replay buffer for the pursuer
        # TODO(Dolev): we add the same reward to both pursuer networks.
        # TODO(Dolev): makes no sense to add com. penalty to TD3 move. net, or collision penalty to PPO com.
        
        if args.opponent_modelling_p:
            if args.PDQN:
                pursuer_agent.store_buffer(mov_action[0], next_pursuer_state,
                                        action, probs, vals,
                                        *opponent_action, uncertainty, pursuer_mediator.loss, # Added evader's true action for opponent modeling loss signal
                                        pursuer_state, rewards['rew_p'], done)
                
            else:                
                pursuer_agent.store_buffer(mov_action, next_pursuer_state,
                                        action, probs, vals,
                                        *opponent_action, uncertainty, pursuer_mediator.loss, # Added evader's true action for opponent modeling loss signal
                                        pursuer_state, rewards['rew_p'], done)


        else:
            pursuer_agent.store_buffer(mov_action, next_pursuer_state,
                                    action, probs, vals,
                                    pursuer_state, rewards['rew_p'], done)
            
        
        
        if args.opponent_modelling_e:
            evader_agent.store_buffer(mov_action_e, next_evader_state,
                                        *opponent_action_e, uncertainty_e, evader_mediator.loss, # Added evader's true action for opponent modeling loss signal
                                        evader_state, rewards['rew_e'], done)
        else:
            if args.Evader_MHPPO:  
                evader_agent.store_buffer(mov_action_e, 0,
                                          0, probs, value,
                                          evader_state, rewards['rew_e'], done)  
            elif args.Evader_PDQN:
                evader_agent.store_buffer(mov_action_e, next_evader_state, 
                                          0, 0, 0,
                                          evader_state, rewards['rew_e'], done)   
           
            else:
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
            # import pandas as pd
            # import time
            
            # t_start = time.time()
            # Update pursuer and evader agents
            
            pursuer_agent.train()
            evader_agent.train()
            
            # t_end = time.time()
            # pd.DataFrame([[t,t_end - t_start]],columns=['timestep','time']).to_csv('/home/afg0547/UUV_DG/self-supervised/UUV_DG/train/times.csv', 
            #                                                                        mode='a',index=False, header=False)

        if done:
            
            if args.opponent_modelling_p:
                if args.PDQN:
                    print(
                        f"Total T: {t + 1} Episode Num: {episode_num + 1}, Episode T: {episode_timestep} Rew_p: {episode_rew_p:.4f}, Rew_e: {episode_rew_e:.4f}, pursuer_mediator.loss: {pursuer_mediator.loss[0]:.4f},")
                else:
                    print(
                        f"Total T: {t + 1} Episode Num: {episode_num + 1}, Episode T: {episode_timestep} Rew_p: {episode_rew_p:.4f}, Rew_e: {episode_rew_e:.4f}, pursuer_mediator.loss: {pursuer_mediator.loss[0]:.4f}, ")
            else:
                print(
                    f"Total T: {t + 1} Episode Num: {episode_num + 1}, Episode T: {episode_timestep} Rew_p: {episode_rew_p:.4f}, Rew_e: {episode_rew_e:.4f}, ")

            global_state, _ = env.reset()

            pursuer_state = env._get_pursuer_state(global_state)
            evader_state = env._get_evader_state(global_state)

            pursuer_mediator.reset(env)
            evader_mediator.reset(env)

            pursuer_state = pursuer_mediator.process_observation(pursuer_state)
            evader_state = evader_mediator.process_observation(evader_state)

            pursuer_agent.clear_history()
            evader_agent.clear_history()

            
            if t > args.start_time_steps:
                if (args.multiheadPPO==False) & (args.PDQN==False) & (args.HyAR==False) & (args.MPDQN==False) & (args.Evader_MHPPO==False):
                    pursuer_agent.movement_agent.expl_noise *= args.expl_noise_decay
                    pursuer_agent.movement_agent.expl_noise = max(pursuer_agent.movement_agent.expl_noise,
                                                                args.min_expl_noise)
                    if not args.Evader_PDQN:
                        evader_agent.movement_agent.expl_noise *= args.expl_noise_decay
                        evader_agent.movement_agent.expl_noise = max(evader_agent.movement_agent.expl_noise, args.min_expl_noise)
            
            # Update counters
            episode_rew_p = 0
            episode_rew_e = 0
            episode_timestep = 0
            episode_num += 1

