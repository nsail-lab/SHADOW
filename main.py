import argparse
import copy
import os
import torch
import numpy as np
import json

from mediators.DummyMediator import DummyMediator

from mediators.SelfSupervisedStateMediator import SelfSupervisedStateMediator
from mediators.TransparentMediator import TransparentMediator
# from model.ComTD3 import ComTD3Agent
# from model.TD3 import TD3Agent
from model.TD3_original import TD3Agent
from model.EvaderAgent import EvaderAgent
# from model.TD3_LSTM import TD3Agent
from model.PursuerAgent import PursuerAgent
from model.PursuerAgent_MHPPO import PursuerAgent_MHPPO
from model.PursuerAgent_PDQN import PursuerAgent_PDQN
from model.PursuerAgent_HyAR import PursuerAgent_HyAR
import train.com_no_evader as com_no_evader
import train.train_rich as train_rich

import train.com_with_evader as com_with_evader
from model.StateThinkingPursuer import StateThinkingPursuerAgent
from model.StateThinkingPursuer_noLSTM import StateThinkingPursuerAgent as StateThinkingPursuerAgent_noLSTM

from model.StateThinkingEvader import StateThinkingEvaderAgent
from model.StateThinkingEvader_noLSTM import StateThinkingEvaderAgent as StateThinkingEvaderAgent_noLSTM
from model.StateThinkingPursuer_LIAM import StateThinkingPursuerAgentLIAM
from model.StateThinkingEvader_LIAM import StateThinkingEvaderAgentLIAM

from model.StateThinkingPursuer_MHPPO import StateThinkingPursuerAgent_MHPPO
from model.StateThinkingPursuerAgent_PDQN import StateThinkingPursuerAgent_PDQN
from model.Pursuer_MPDQN import PursuerAgent_MPDQN

from model.EvaderAgent_MHPPO import EvaderAgent_MHPPO
from model.EvaderAgent_PDQN import EvaderAgent_PDQN
# from model.StateThinkingEvader_PDQN import StateThinkingEvaderAgent_PDQN


#from train.com_no_evader import train
from train.train_simple import rl_train

from datetime import datetime

from environments.rich_env_tfeats import DifferentialGameEnvironment as RichPursuerEvaderEnv_Teats
from environments.SimplePursuerEnv import DifferentialGameEnvironment as SimplePursuerEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--num_pursuers", default=1, type=int)
    parser.add_argument("--map_width", default=1, type=float)
    parser.add_argument("--map_height", default=1, type=float)
    parser.add_argument("--pursuer_speed", default=7.7e-3, type=float)  # 15knot~7.7m/s
    parser.add_argument("--evader_speed", default=7.7e-3, type=float)   
    parser.add_argument("--pursuer_max_acceleration", default=0.9*np.pi, type=float)  # 0.9pi rad/sec
    parser.add_argument("--evader_max_acceleration", default=0.9*np.pi, type=float)  
    parser.add_argument("--r_min", default=0.025, type=float)  
    parser.add_argument("--r_shoot", default=0.1, type=float)  # lambd exponential parameter
    parser.add_argument("--dt", default=1, type=float)  
    parser.add_argument("--beta", default=1_000, type=float)
    parser.add_argument("--communication_penalty", default=0, type=float)  
    parser.add_argument("--time_penalty", default=0.5, type=float)
    parser.add_argument("--hit_boundary_penalty",default=10, type=float)
    parser.add_argument("--lat_acceleration_penalty",default=0.5, type=float)
    parser.add_argument("--shooting_penalty",default=100,type=float)
    parser.add_argument("--distance_penalty",default=100,type=float)
    parser.add_argument("--max_steps", default=1_000, type=int)
    parser.add_argument("--communication", default=True, type=bool)
    parser.add_argument("--smart_e", default=True, type=bool)
    parser.add_argument("--probabilistic", default=True, type=bool)
    parser.add_argument("--lambd", default=2, type=int)
    parser.add_argument("--generalization", default=True, type=bool)
    parser.add_argument("--noise_std",default=0, type=float)
    parser.add_argument("--random_noise", default=False, type=bool)
    
    # TD3
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float) # also for PPO
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_delay", default=5, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--replay_buffer_size", default=1_000_000, type=int)

    # PPO
    parser.add_argument("--gae_lambda", default=0.95, type=float)
    parser.add_argument("--policy_clip", default=0.2, type=float)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--history_len", default=1, type=int)
    
    
    parser.add_argument("--multiheadPPO", default=False, type=bool)
    parser.add_argument("--PDQN", default=False, type=bool)
    parser.add_argument("--HyAR", default=False, type=bool)
    parser.add_argument("--MPDQN", default=False, type=bool)
    parser.add_argument("--LIAM", default=False, type=bool)
    
    parser.add_argument("--Evader_MHPPO", default=False, type=bool)
    parser.add_argument("--Evader_LIAM", default=False, type=bool)
    parser.add_argument("--Evader_PDQN", default=False, type=bool)

    # Training
    parser.add_argument("--start_time_steps", default=15_000, type=int) 
    parser.add_argument("--max_timesteps", default=6_000_000, type=int)
    parser.add_argument("--batch_size", default=32, type=int) 
    parser.add_argument("--save_model", default=True, type=bool)
    parser.add_argument("--eval_freq", default=200, type=int)
    parser.add_argument("--eval_episodes", default=50, type=int)

    parser.add_argument("--seed", default=3, type=int)
    parser.add_argument("--load_seed", default=1, type=int)
    parser.add_argument("--load_model", default=False, type=bool) # not tested
    parser.add_argument("--load_model_dir", default='', type=str) # not tested

    parser.add_argument("--min_expl_noise", default=0.1, type=float)
    parser.add_argument("--expl_noise_decay", default=1, type=float)
    parser.add_argument("--cuda", default="0", type=str)

    parser.add_argument("--random_communication", default=False, type=bool)
    parser.add_argument("--random_communication_th", default=0, type=float)
    parser.add_argument("--communication_period", default=40, type=float)
    
    parser.add_argument("--opponent_modelling_p", default=True, type=bool)
    parser.add_argument("--opponent_modelling_e", default=True, type=bool)
    parser.add_argument("--use_LSTM_p", default=True, type=bool)
    parser.add_argument("--use_LSTM_e", default=True, type=bool)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if (args.save_model):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f'./models/{timestamp}'
        os.mkdir(folder)
        args.save_model_path = folder

        # Convert the parsed arguments to a dictionary
        params = vars(args)

        # Save the dictionary to a JSON file
        with open(os.path.join(folder,"params.json"), "w") as f:
            json.dump(params, f, indent=4)
    else:
        args.file_name = None
        args.save_model_path = None


    # Initialize environment
    env_kwargs = {
        "num_pursuers": args.num_pursuers,
        "map_size": (args.map_width, args.map_height),
        "pursuer_speed": args.pursuer_speed,
        "evader_speed": args.evader_speed,
        "pursuer_max_acceleration": args.pursuer_max_acceleration,
        "evader_max_acceleration": args.evader_max_acceleration,
        "r_min": args.r_min,
        "r_shoot": args.r_shoot,
        "dt": args.dt,
        "beta": args.beta,
        "lat_acceleration_penalty": args.lat_acceleration_penalty,
        "communication_penalty": args.communication_penalty,
        "distance_penalty": args.distance_penalty,
        'time_penalty': args.time_penalty,
        "hit_boundary_penalty": args.hit_boundary_penalty,
        "shooting_penalty": args.shooting_penalty,
        "max_steps": args.max_steps,
        "probabilistic": args.probabilistic,
        "lambd": args.lambd,
        "generalization": args.generalization,
        "random_communication": args.random_communication,
        "random_communication_th": args.random_communication_th,
        "communication_period": args.communication_period,
        "noise_std": args.noise_std
    }
    
    print('[DEBUG] - env_kwargs: ', env_kwargs)
    if args.communication:
        env = RichPursuerEvaderEnv_Teats(**env_kwargs) #RichPursuerEvaderEnv(**env_kwargs)

    else:
        env = SimplePursuerEnv(**env_kwargs)

    # TODO(Dolev): retrieve from environment, more robust
    evader_state_dim = 19 
    pursuer_state_dim = 19 
    action_dim = 1
    
    print('[DEBUG] - ', args.opponent_modelling_p, args.opponent_modelling_e)
    if args.communication:
        model_kwargs = {
            "state_dim": pursuer_state_dim,
            "action_dim": action_dim,
            "max_action": args.pursuer_max_acceleration,
            "hidden_dim": args.hidden_dim,
            "discount": args.discount,
            "tau": args.tau,
            "policy_noise": args.policy_noise,
            "noise_clip": args.noise_clip,
            "policy_delay": args.policy_delay,
            "expl_noise": args.expl_noise,

            "gae_lambda": args.gae_lambda,
            "policy_clip": args.policy_clip,
            "alpha": args.lr,
            "history_len": args.history_len,

            "replay_buffer_size": args.replay_buffer_size,
            "batch_size": args.batch_size,

            "cuda":args.cuda
        }
        pursuer_model_kwargs = copy.deepcopy(model_kwargs)
        
        if args.multiheadPPO:
        # pursuer_agent = PursuerAgent(**pursuer_model_kwargs)
            if args.opponent_modelling_p:
                pursuer_agent = StateThinkingPursuerAgent_MHPPO(**pursuer_model_kwargs) 
            else:
                pursuer_agent = PursuerAgent_MHPPO(**pursuer_model_kwargs) # 
        elif args.PDQN:
            if args.opponent_modelling_p:
                pursuer_agent = StateThinkingPursuerAgent_PDQN(**pursuer_model_kwargs)
            else:
                pursuer_agent = PursuerAgent_PDQN(pursuer_state_dim,args.pursuer_max_acceleration, cuda = args.cuda) #
        elif args.MPDQN:
            pursuer_agent = PursuerAgent_MPDQN(**pursuer_model_kwargs)
        elif args.HyAR:
            pursuer_agent = PursuerAgent_HyAR(**pursuer_model_kwargs) #
        elif args.LIAM:
            pursuer_agent = StateThinkingPursuerAgentLIAM(**pursuer_model_kwargs) 
            
        else:
            if args.opponent_modelling_p:
                if args.use_LSTM_p:
                    pursuer_agent = StateThinkingPursuerAgent(**pursuer_model_kwargs)
                else:
                    pursuer_agent = StateThinkingPursuerAgent_noLSTM(**pursuer_model_kwargs)
            else:
                pursuer_agent = PursuerAgent(**pursuer_model_kwargs) # 
        print(pursuer_agent)
    else:
        model_kwargs = {
            "state_dim": pursuer_state_dim,
            "action_dim": action_dim,
            "discount": args.discount,
            "tau": args.tau,
            "policy_delay": args.policy_delay,
            "expl_noise": args.expl_noise,
            "replay_buffer_size": args.replay_buffer_size,
            "batch_size": args.batch_size,
            "cuda": args.cuda
        }
        pursuer_model_kwargs = copy.deepcopy(model_kwargs)
        pursuer_model_kwargs["max_action"] = args.pursuer_max_acceleration
        
        pursuer_agent = TD3Agent(**pursuer_model_kwargs)


    
    if args.smart_e:
        evader_kwargs = {
            "state_dim": evader_state_dim,
            "action_dim": action_dim,
            "max_action": args.pursuer_max_acceleration,
            "hidden_dim": args.hidden_dim,
            "discount": args.discount,
            "tau": args.tau,
            "policy_noise": args.policy_noise,
            "noise_clip": args.noise_clip,
            "policy_delay": args.policy_delay,
            "expl_noise": args.expl_noise,
            "history_len": args.history_len,
            "replay_buffer_size": args.replay_buffer_size,
            "batch_size": args.batch_size,
            "cuda": args.cuda
        }      
        if args.opponent_modelling_e:
            if args.Evader_LIAM:
                evader_agent = StateThinkingEvaderAgentLIAM(**evader_kwargs)
            else:
                if args.use_LSTM_e:
                    evader_agent = StateThinkingEvaderAgent(**evader_kwargs) # 
                else:
                    evader_agent = StateThinkingEvaderAgent_noLSTM(**evader_kwargs) #
        else:
            if args.Evader_MHPPO:
                evader_agent = EvaderAgent_MHPPO(**evader_kwargs)
            elif args.Evader_PDQN:
                evader_agent = EvaderAgent_PDQN(evader_state_dim,args.evader_max_acceleration, cuda = args.cuda)
            else:
                evader_agent = EvaderAgent(**evader_kwargs)
        
        print(evader_agent)
        
        

    if args.load_model:
        print('[DEBUG] - loading previous model...')
        pursuer_agent.load(args.load_model_dir)
        if args.smart_e:
            evader_agent.load(args.load_model_dir)
    

    # Train model
    train_kwargs = {
             'min_expl_noise': args.min_expl_noise,
             'expl_noise_decay': args.expl_noise_decay,
             'start_time_steps': args.start_time_steps,
             'max_timesteps': args.max_timesteps,
             'eval_freq': args.eval_freq,
             'max_steps': args.max_steps,
             'eval_episodes': args.eval_episodes,
             'save_model_path': args.save_model_path,
    }

    if args.communication:
        if args.smart_e:
            print('Training: Rich + Communication + Smart Evader')
            if args.opponent_modelling_p:
                pursuer_mediator = SelfSupervisedStateMediator(opponent="evader")
            else:
                pursuer_mediator = DummyMediator()
                
            if args.opponent_modelling_e:
                evader_mediator = SelfSupervisedStateMediator(opponent="pursuer")
            else:
                evader_mediator = DummyMediator()
                
            train_rich.train(pursuer_agent, evader_agent,
                             pursuer_mediator, evader_mediator,
                             env, args)

        else:
            print('Training: Communication + No Evader')
            com_no_evader.train(pursuer_agent, DummyMediator(), env, args)

    else:
        print('[DEBUG] - train_kwargs: ', train_kwargs)
        rl_train(pursuer_agent, DummyMediator(), env, **train_kwargs)
