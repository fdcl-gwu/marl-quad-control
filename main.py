import os, sys, torch, copy, gym_rotor
# os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-43983c88-ad09-55fa-a5f4-884dffcb799d"
import gymnasium as gym
from datetime import datetime
import args_parse

import numpy as np
from gym_rotor.envs.quad_utils import *
from gym_rotor.envs.quad_eIx_utils import *
from gym_rotor.wrappers.decoupled_yaw_wrapper import DecoupledWrapper
from algos.replay_buffer import ReplayBuffer
from algos.matd3 import MATD3
from algos.td3 import TD3

# Create directories:    
os.makedirs("./models") if not os.path.exists("./models") else None 
os.makedirs("./results") if not os.path.exists("./results") else None

class Learner:
    def __init__(self, args, framework, seed):
    
        # Make OpenAI Gym environment:
        self.args = args
        self.framework = framework
        self.total_timesteps = 0
        if self.framework in ("DTDE", "CTDE"):
            """--------------------------------------------------------------------------------------------------
            | Agents  | Observations           | obs_dim | Actions:       | act_dim | Rewards                   |
            | #agent1 | {ex, ev, b3, w12, eIx} | 15      | {f_total, tau} | 4       | f(ex, ev, eb3, ew12, eIx) |
            | #agent2 | {b1, W3, eIb1}         | 5       | {M3}           | 1       | f(eb1, eW3, eIb1)         |
            --------------------------------------------------------------------------------------------------"""
            self.env = DecoupledWrapper()
            self.args.N = 2 # The number of agents
            self.args.obs_dim_n = [15, 5]   
            self.args.action_dim_n = [4, 1] 
        elif self.framework == "SARL":
            """--------------------------------------------------------------------------------------------------
            | Agents  | Observations           | obs_dim | Actions:       | act_dim | Rewards                   |
            | #agent1 | {ex, ev, R, eW, eIx}   | 21      | {T1,T2,T3,T4}  | 4       | f(ex, ev, eb1, eW, eIx)   |
            --------------------------------------------------------------------------------------------------"""
            self.env = gym.make("Quad-v1", render_mode="human")
            self.args.N = 1 # The number of agents
            self.args.obs_dim_n = [21, 21]
            self.args.action_dim_n = [4, 4] 
        
        # Set seed for random number generators:
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.action_space.seed(self.seed)
        self.env.observation_space.seed(self.seed)
        
        # Initialize N agents:
        if self.framework == "CTDE":
            self.agent_n = [MATD3(args, agent_id) for agent_id in range(args.N)]
        elif self.framework in ("SARL", "DTDE"):
            self.agent_n = [TD3(args, agent_id) for agent_id in range(args.N)]
        self.args.noise_std_decay = (args.explor_noise_std_init - args.explor_noise_std_min) / args.explor_noise_decay_steps
        self.explor_noise_std = self.args.explor_noise_std_init # Initialize explor_noise_std

        # Initialize replay buffer:
        self.replay_buffer = ReplayBuffer(self.args)
        
        # Load trained models and optimizer parameters:
        if args.test_model == True:
            for agent_id in range(self.args.N):
                self.agent_n[agent_id].load(self.framework, 2890_000, agent_id, self.seed) 
                # self.agent_n[agent_id].load_solved_model(self.framework, 2250_000, agent_id, self.seed) 


    def train_policy(self):
        # Evaluate policy:
        self.eval_policy()
        sys.exit("The trained agent has been test!") if args.test_model == True else None

        # Setup loggers:
        log_step_path = os.path.join("./results", "log_step_seed_"+str(self.seed)+".txt")   
        log_eval_path = os.path.join("./results", "log_eval_seed_"+str(self.seed)+".txt")
        log_step = open(log_step_path,"w+") # Total timesteps vs. Total reward
        log_eval = open(log_eval_path,"w+") # Total timesteps vs. Evaluated average reward

        # Initialize environment:
        obs_n, done_episode = self.env.reset(env_type='train', seed=self.seed), False
        b1d = self.env.b1d
        max_total_reward = args.max_steps*0.9 # to save best models
        episode_timesteps, episode_reward = 0, 0

        # Training loop:
        for self.total_timesteps in range(int(self.args.max_timesteps)):
            episode_timesteps += 1

            # Each agent selects actions based on its own local observations w/ exploration noise:
            act_n = [agent.choose_action(obs, explor_noise_std=self.explor_noise_std) for agent, obs in zip(self.agent_n, obs_n)]
            action = np.concatenate((act_n), axis=None)

            # Perform actions:
            obs_next_n, r_n, done_n, _, _ = self.env.step(copy.deepcopy(action))
            eX = np.round(obs_next_n[0][0:3]*self.env.x_lim, 5) # position error [m]
            if self.framework in ("DTDE", "CTDE"):
                eR = ang_btw_two_vectors(obs_next_n[1][0:3], b1d) # heading error [rad]
            elif self.framework == "SARL":
                eR = ang_btw_two_vectors(obs_next_n[0][6:9], b1d) # heading error [rad]

            # Episode termination:
            if episode_timesteps == self.args.max_steps: # Episode terminated!
                done_episode = True
                done_n[0] = True if (abs(eX) <= 0.05).all() else False # Problem is solved!
                if self.framework in ("DTDE", "CTDE"):
                    done_n[1] = True if abs(eR) <= 0.05 else False # Problem is solved!
        
            # Store a set of transitions in replay buffer:
            self.replay_buffer.store_transition(obs_n, act_n, r_n, obs_next_n, done_n)
            obs_n = obs_next_n
            episode_reward += sum(r_n)/self.args.N
            self.total_timesteps += 1

            # Decay explor_noise_std:
            if self.args.use_explor_noise_decay:
                self.explor_noise_std = self.explor_noise_std - self.args.noise_std_decay if self.explor_noise_std - self.args.noise_std_decay > self.args.explor_noise_std_min else self.args.explor_noise_std_min

            # Train agent after collecting sufficient data:
            if self.total_timesteps > self.args.start_timesteps:
                # Train each agent individually:
                for agent_id in range(self.args.N):
                    self.agent_n[agent_id].train(self.replay_buffer, self.agent_n, self.env)

            # Evaluate policy:
            if self.total_timesteps % self.args.eval_freq == 0 and self.total_timesteps > self.args.start_timesteps:
                eval_reward = self.eval_policy()

                # Logging updates:
                if self.framework in ("DTDE", "CTDE"):
                    log_eval.write('{}\t {}\n'.format(self.total_timesteps, eval_reward))
                elif self.framework == "SARL":
                    log_eval.write('{}\t {}\n'.format(self.total_timesteps, eval_reward[0]))
                log_eval.flush()

                # Save best model:
                if eval_reward > max_total_reward:
                    max_total_reward = eval_reward
                    for agent_id in range(self.args.N):
                        self.agent_n[agent_id].save_model(self.framework, self.total_timesteps, agent_id, self.seed)
               
            # If done_episode:
            if any(done_n) == True or done_episode == True:
                if self.framework in ("DTDE", "CTDE"):
                    print(f"total_timestpes: {self.total_timesteps+1}, time_stpes: {episode_timesteps}, reward: {episode_reward:.3f}, eX: {eX}, eR: {eR:.3f}")
                elif self.framework == "SARL":
                    print(f"total_timestpes: {self.total_timesteps+1}, time_stpes: {episode_timesteps}, reward: {episode_reward[0]:.3f}, eX: {eX}, eR: {eR:.3f}")

                # Log data:
                if self.total_timesteps >= self.args.start_timesteps:
                    if self.framework in ("DTDE", "CTDE"):
                        log_step.write('{}\t {}\n'.format(self.total_timesteps, episode_reward))
                    elif self.framework == "SARL":
                        log_step.write('{}\t {}\n'.format(self.total_timesteps, episode_reward[0]))
                    log_step.flush()
                
                # Reset environment:
                obs_n, done_episode = self.env.reset(env_type='train', seed=self.seed), False
                episode_timesteps, episode_reward = 0, 0

        # Close environment:
        self.env.close()


    def eval_policy(self):
        # Make OpenAI Gym environment:
        if self.framework in ("DTDE", "CTDE"):
            """--------------------------------------------------------------------------------------------------
            | Agents  | Observations           | obs_dim | Actions:       | act_dim | Rewards                   |
            | #agent1 | {ex, ev, b3, w12, eIx} | 15      | {f_total, tau} | 4       | f(ex, ev, eb3, ew12, eIx) |
            | #agent2 | {b1, W3, eIb1}         | 5       | {M3}           | 1       | f(eb1, eW3, eIb1)         |
            --------------------------------------------------------------------------------------------------"""
            eval_env = DecoupledWrapper()
        elif self.framework == "SARL":
            """--------------------------------------------------------------------------------------------------
            | Agents  | Observations           | obs_dim | Actions:       | act_dim | Rewards                   |
            | #agent1 | {ex, ev, R, eW, eIx}   | 21      | {T1,T2,T3,T4}  | 4       | f(ex, ev, eb1, eW, eIx)   |
            --------------------------------------------------------------------------------------------------"""
            eval_env = gym.make("Quad-v1", render_mode="human")

        # Fixed seed is used for the eval environment.
        seed = 123
        eval_env.action_space.seed(seed)
        eval_env.observation_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Save solved model:
        success_count = [] if args.save_model else None

        eval_reward = 0.
        print("---------------------------------------------------------------------------------------------------------------------")
        for num_eval in range(self.args.num_eval):
            episode_reward, episode_timesteps = 0, 0

            # Reset envs:
            obs_n = eval_env.reset(env_type='eval', seed=self.seed)

            # Goal state:
            xd = np.array([0.,0.,0.])/eval_env.x_lim 
            xd_dot = np.array([0.,0.,0.])/eval_env.v_lim  
            Wd = np.array([0.,0.,0.])/eval_env.W_lim 
            b1d = np.array([1.,0.,0.]) # desired heading direction
            b3d = np.array([0.,0.,1.])

            # Data save:
            act_list, obs_list, cmd_list = [], [], [] if args.save_log else None

            # Evaluation loop:
            for _ in range(self.args.max_steps):
                episode_timesteps += 1
                # Actions w/o exploration noise:
                act_n = [agent.choose_action(obs, explor_noise_std=0) for agent, obs in zip(self.agent_n, obs_n)] 
                action = np.concatenate((act_n), axis=None)

                # Perform actions:
                obs_next_n, r_n, done_n, state_next, _ = eval_env.step(copy.deepcopy(action))

                # Cumulative rewards:
                episode_reward += sum(r_n)/self.args.N
                obs_n = obs_next_n

                # Save data:
                if args.save_log:
                    act_list.append(action)
                    obs_list.append(state_next)
                    cmd_list.append(np.concatenate((xd, xd_dot, b1d, b3d, Wd), axis=None))

                # Episode termination:
                if any(done_n) or episode_timesteps == args.max_steps:
                    eX = np.round(obs_next_n[0][0:3]*eval_env.x_lim, 5) # position error [m]
                    if self.framework in ("DTDE", "CTDE"):
                        eR = ang_btw_two_vectors(obs_next_n[1][0:3], b1d) # heading error [rad]
                        print(f"eval_iter: {num_eval+1}, time_stpes: {episode_timesteps}, episode_reward: {episode_reward:.3f}, eX: {eX}, eR: {eR:.3f}")
                    elif self.framework == "SARL":
                        eR = ang_btw_two_vectors(obs_next_n[0][6:9], b1d) # heading error [rad]
                        print(f"eval_iter: {num_eval+1}, time_stpes: {episode_timesteps}, episode_reward: {episode_reward[0]:.3f}, eX: {eX}, eR: {eR:.3f}")
                    success = True if (abs(eX) <= 0.05).all() else False
                    success_count.append(success)
                    break

            eval_reward += episode_reward
            # Save data:
            if args.save_log:
                min_len = min(len(act_list), len(obs_list), len(cmd_list))
                log_data = np.column_stack((act_list[-min_len:], obs_list[-min_len:], cmd_list[-min_len:]))
                header = "Actions and States\n"
                header += "action[0], ..., state[0], ..., command[0], ..." 
                time_now = datetime.now().strftime("%m%d%Y_%H%M%S") 
                fpath = os.path.join('./results', 'log_' + time_now + '.dat')
                np.savetxt(fpath, log_data, header=header, fmt='%.10f') 

        # Average reward:
        eval_reward = eval_reward / self.args.num_eval
        print("------------------------------------------------------------------------------------------")
        if self.framework in ("DTDE", "CTDE"):
            print(f"total_timesteps: {self.total_timesteps} \t eval_reward: {eval_reward:.3f} \t explor_noise_std: {self.explor_noise_std}")
        elif self.framework == "SARL":
            print(f"total_timesteps: {self.total_timesteps} \t eval_reward: {eval_reward[0]:.3f} \t explor_noise_std: {self.explor_noise_std}")
        print("------------------------------------------------------------------------------------------")

        # Save solved model:
        if all(i == True for i in success_count) and args.save_model == True: # Problem is solved
            for agent_id in range(self.args.N):
                self.agent_n[agent_id].save_solved_model(self.framework, self.total_timesteps, agent_id, self.seed)

        return eval_reward
        
        
if __name__ == '__main__':
    # Hyperparameters:
    parser = args_parse.create_parser()
    args = parser.parse_args()

    # Show information:
    print("---------------------------------------------------------------------------------------------------------------------")
    print("Framework:", args.framework_id, "| Seed:", args.seed, "| Batch size:", args.batch_size)
    print("gamma:", args.discount, "| lr_a:", args.lr_a, "| lr_c:", args.lr_c, 
          "| Actor hidden dim:", args.actor_hidden_dim, 
          "| Critic hidden dim:", args.critic_hidden_dim)
    print("---------------------------------------------------------------------------------------------------------------------")

    learner = Learner(args, framework=args.framework_id, seed=args.seed)
    learner.train_policy()