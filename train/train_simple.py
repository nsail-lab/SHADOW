from tqdm import tqdm
import numpy as np
import matplotlib#; matplotlib.use("TkAgg")

from mediators.Mediator import Mediator


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

# Training code link: https://github.com/sfujim/TD3/blob/master/main.py

def eval_policy(pursuer_agent, pursuer_mediator: Mediator,
                env, eval_episodes=10, max_steps=1_000):
    avg_reward = 0.0
    for seed in tqdm(range(eval_episodes), total=eval_episodes):
        pursuer_states, evader_state = env.reset(seed=seed)
        pursuer_states = pursuer_mediator.process_observation(pursuer_states)
        steps = 0
        total_reward = 0
        done = False
        while not done and steps < max_steps:
            steps += 1
            pursuer_actions = pursuer_agent.select_action(pursuer_states) # [pursuer_agent.select_action(state) for state in pursuer_states]
            pursuer_states, rewards, termination, truncation, _ = env.step(pursuer_actions)
            pursuer_states = pursuer_mediator.process_observation(pursuer_states)
            done = termination or truncation
            total_reward += np.min(rewards)
            
        avg_reward += total_reward
        #env.history_animation(total_reward, arrow_length=0.05, arrow_width=0.01)
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def rl_train(pursuer_agent,
             pursuer_mediator: Mediator,
             env,
             file_name=None,
             min_expl_noise=0.1,
             expl_noise_decay=0.99,
             start_time_steps=25_000,
             max_timesteps=1_000_000,
             eval_freq=1000,
             max_steps=10_000,
             eval_episodes=10):
    # Evaluate policy before training
    eval_policy(pursuer_agent, pursuer_mediator, env, eval_episodes=eval_episodes, max_steps=max_steps)

    episode_timestep = 0
    episode_num = 0
    episode_reward = 0
    evaluations = []

    pursuer_states, _ = env.reset()
    pursuer_states = pursuer_mediator.process_observation(pursuer_states)

    # Training loop
    for t in range(max_timesteps):
        episode_timestep += 1

        if t < start_time_steps:
            # Select actions for pursuers
            pursuer_actions = pursuer_agent.sample_action() # [pursuer_agent.sample_action() for _ in range(env.num_pursuers)]
        else:
            # Select actions for pursuers
            pursuer_actions = pursuer_agent.select_train_action(pursuer_states) #[pursuer_agent.select_train_action(state) for state in pursuer_states]

        # Execute actions and observe next states and rewards
        next_pursuer_states, rewards, termination, truncation, _ = env.step(pursuer_actions)
        next_pursuer_states = pursuer_mediator.process_observation(next_pursuer_states)
        done = termination or truncation

        # Store experiences in replay buffer for the pursuers
        pursuer_agent.replay_buffer.add(pursuer_states, pursuer_actions, rewards, next_pursuer_states, done)

        pursuer_states = next_pursuer_states
        episode_reward += np.min(rewards)

        if t >= start_time_steps:
            # Update pursuer and evader agents
            pursuer_agent.train()

        if done:
            # import pandas as pd
            # from matplotlib import pyplot as plt
            # if t > start_time_steps and episode_num % eval_freq == 1:
            #     pd.DataFrame(env.evader_actions).plot()
            #     plt.show()
                # env.history_animation(episode_reward, arrow_length=arrow_length, arrow_width=arrow_width)
            # Print episode statistics
            print(f"Total T: {t+1} Episode Num: {episode_num + 1}, Episode T: {episode_timestep} Reward: {episode_reward}, Explor. Noise: {pursuer_agent.expl_noise}")

            # Reset environment and get initial states
            pursuer_states, _ = env.reset()
            pursuer_states = pursuer_mediator.process_observation(pursuer_states)
            if t > start_time_steps:
                pursuer_agent.expl_noise *= expl_noise_decay
                pursuer_agent.expl_noise = max(pursuer_agent.expl_noise, min_expl_noise)
                #pursuer_agent.state_history.clear()
            # Update counters
            episode_reward = 0
            episode_timestep = 0
            episode_num += 1

        # Evaluate episode
        if episode_num % eval_freq == 0 and t > start_time_steps:
            evaluations.append(eval_policy(pursuer_agent, pursuer_mediator,
                                           env,
                                           eval_episodes=eval_episodes,
                                           max_steps=max_steps))
            env.reset()
            episode_num += 1
            if file_name:
                np.save(f"{file_name}", evaluations)
                pursuer_agent.save(f"{file_name}")
    # save model
    if file_name:
        np.save(f"{file_name}", evaluations)
        pursuer_agent.save(f"{file_name}")
