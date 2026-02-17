import copy
from tqdm import tqdm
import numpy as np
import os

from mediators.Mediator import Mediator


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(pursuer_agent, pursuer_mediator: Mediator, env, eval_episodes=10, max_steps=1_000, eval=False, verbose=False):
    avg_reward = 0.
    meta = {}
    for seed in tqdm(range(eval_episodes), total=eval_episodes):
        obs, _ = env.reset(seed)
        pursuer_state = pursuer_mediator.process_observation(copy.deepcopy(obs))

        steps = 0
        total_reward = 0
        done = False

        meta[seed] = {'states': [], 'rewards': [], 'infos': [], 'n_steps': 0, 'truncation': False, 'termination': False}
        meta[seed]['states'].append(obs)

        while not done and steps < max_steps:
            steps += 1

            mov_action, com_action = pursuer_agent.select_action(pursuer_state)

            actions = [mov_action[0], com_action]

            new_obs, rewards, termination, truncation, info = env.step(actions)
            meta[seed]['states'].append(new_obs)
            meta[seed]['rewards'].append(rewards)
            meta[seed]['infos'].append(info)
            done = termination or truncation
            pursuer_state = pursuer_mediator.process_observation(copy.deepcopy(new_obs))
            total_reward += np.min(rewards)
            #print(total_reward)

        pursuer_agent.clear_history()
        
        meta[seed]['n_steps'] = steps
        meta[seed]['termination'] = termination
        meta[seed]['truncation'] = truncation
        avg_reward += total_reward
        if verbose:
            print(f'[DEBUG] - episode {seed} | reward {total_reward} | n_steps {steps} ')
        #env.history_animation(total_reward, arrow_length=0.05, arrow_width=0.01)
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    if eval:
        return avg_reward,meta
    else:
        return avg_reward

def train(pursuer_agent, pursuer_mediator: Mediator, env, args):
    episode_timestep = 0
    episode_num = 0
    episode_reward = 0
    evaluations = []
    obs, _ = env.reset()

    pursuer_state = pursuer_mediator.process_observation(copy.deepcopy(obs))

    for t in range(args.max_timesteps):
        # Evaluate episode
        if episode_num % args.eval_freq == 0 and t > args.start_time_steps:
            avg_reward = eval_policy(pursuer_agent,
                                     env,
                                     eval_episodes=args.eval_episodes,
                                     max_steps=args.max_steps)
            evaluations.append(avg_reward)
            env.reset()
            episode_num += 1
            if (args.save_model):
                save_dir = os.path.join(args.save_model_path,str(episode_num))
                os.mkdir(save_dir)
                with open(os.path.join(save_dir,'avg_reward.txt'),'w') as fout:
                    fout.write(str(avg_reward))
                    
                pursuer_agent.save(save_dir)

                np.save(os.path.join(args.save_model_path,'evalutions.npy'), evaluations)

        episode_timestep += 1
        if t < args.start_time_steps:
            # Select actions for pursuers
            mov_action, com_action = pursuer_agent.sample_action(pursuer_state)
        else:
            # Select actions for pursuers
            mov_action, com_action = pursuer_agent.select_train_action(pursuer_state)

        action, probs, vals = com_action
        
        # Execute actions and observe next states and rewards
        actions = np.squeeze(np.array([mov_action[0], action]))

        new_obs, rewards, termination, truncation, info = env.step(actions)

        done = termination or truncation
        next_pursuer_state = pursuer_mediator.process_observation(copy.deepcopy(new_obs))

        # Store experiences in replay buffer for the pursuer
        pursuer_agent.store_buffer(mov_action, next_pursuer_state,
                                   action, probs, vals,
                                   pursuer_state, rewards, done)
        
        pursuer_state = next_pursuer_state
        episode_reward += np.min(rewards)

        if t >= args.start_time_steps:
            # Update pursuer and evader agents
            pursuer_agent.train()

        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num + 1}, Episode T: {episode_timestep} Reward: {episode_reward}, Explor. Noise: {pursuer_agent.movement_agent.expl_noise}")

            obs, _ = env.reset()

            pursuer_agent.clear_history()
            pursuer_state = pursuer_mediator.process_observation(copy.deepcopy(obs))

            if t > args.start_time_steps:
                pursuer_agent.movement_agent.expl_noise *= args.expl_noise_decay
                pursuer_agent.movement_agent.expl_noise = max(pursuer_agent.movement_agent.expl_noise, args.min_expl_noise)
            
            # Update counters
            episode_reward = 0
            episode_timestep = 0
            episode_num += 1

            