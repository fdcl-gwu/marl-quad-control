import os, sys, torch, copy
sys.path.append('../')
import gym_rotor
import gymnasium as gym
from datetime import datetime
import args_parse

import numpy as np
from numpy import random
from gym_rotor.envs.quad_utils import *
from gym_rotor.wrappers.decoupled_yaw_wrapper import DecoupledWrapper
from gym_rotor.wrappers.coupled_yaw_wrapper import CoupledWrapper
from utils.trajectory_generation import TrajectoryGeneration
from algos.replay_buffer import ReplayBuffer
from algos.matd3 import MATD3
from algos.td3 import TD3
from utils.utils import *

# Create directories:    
os.makedirs("./models") if not os.path.exists("./models") else None 
os.makedirs("./results") if not os.path.exists("./results") else None

class Learner:
    def __init__(self, args, framework, seed):
        # PD control gains:
        self.kX = 9.0*np.diag([1.0, 1.0, 2.0]) # position gains  
        self.kV = 5.0*np.diag([1.0, 1.0, 1.5]) # velocity gains   
        self.kR = 2.0*np.diag([1.0, 1.0, 0.5]) # attitude gains 
        self.kW = 2.0*np.diag([0.25, 0.25, 0.2]) # angular velocity gains 
		
        # Integral control:
        self.use_integral = True
        self.sat_sigma = 3.5
        self.eIX = IntegralErrorVec3() # Position integral error
        self.eIR = IntegralErrorVec3() # Attitude integral error
        self.eIX.set_zero() # Set all integrals to zero
        self.eIR.set_zero()
        # I control gains:
        self.kIX = 4.0*np.diag([1.0, 1.0, 1.4])  # Position integral gains
        self.kIR = 0.02*np.diag([1.0, 1.0, 0.7]) # Attitude integral gain

        # Make OpenAI Gym environment:
        self.args = args
        self.framework, self.args.N = 'SARL', 1

        # Set seed for random number generators:
        self.seed = seed
        

    def eval_policy(self):
        # Make OpenAI Gym environment:
        eval_env = CoupledWrapper()
        self.eval_max_steps = self.args.eval_max_steps/eval_env.dt 
        self.trajectory = TrajectoryGeneration(eval_env)
        self.x_lim, self.v_lim, self.W_lim = eval_env.x_lim, eval_env.v_lim, eval_env.W_lim
        self.dt, self.m, self.J, self.g = eval_env.dt, eval_env.m, eval_env.J, eval_env.g
        self.scale_act, self.avrg_act = eval_env.scale_act, eval_env.avrg_act

        # Fixed seed is used for the eval environment.
        seed = 123
        eval_env.action_space.seed(seed)
        eval_env.observation_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Save rewards:
        eval_reward = [0.]
        benchmark_reward = 0. # Reward for benchmark

        print("---------------------------------------------------------------------------------------------------------------------")
        for num_eval in range(self.args.num_eval):
            # Set mode for generating trajectory:
            mode = 5
            """ Mode List -----------------------------------------------
            0 or 1: idle and warm-up (approach to xd = [0,0,0])
            2: take-off
            3: landing
            4: stay (hovering)
            5: circle
            ----------------------------------------------------------"""
            self.trajectory.mark_traj_start() # reset trajectory

            # Data save:
            act_list, obs_list, cmd_list = [], [], [] if args.save_log else None

            # Reset envs, timesteps, and reward:
            obs_n = eval_env.reset(env_type='eval', seed=self.seed)
            episode_timesteps, episode_reward = 0, [0.]
            episode_benchmark_reward = 0.

            # Evaluation loop:
            for _ in range(int(self.eval_max_steps)):
                episode_timesteps += 1

                # Generate trajectory:
                state = eval_env.get_current_state()
                xd, vd, b1d, b3d, Wd = self.trajectory.get_desired(state, mode)
                eval_env.set_goal_pos(xd, b1d)
                error_obs_n, error_state = self.trajectory.get_error_state(obs_n, self.framework)

                # Actions w/o exploration noise:
                action = np.concatenate(self.geometricTrackingController(state), axis=None)

                # Perform actions:
                obs_next_n, r_n, done_n, _, _ = eval_env.step(copy.deepcopy(action))
                state_next = eval_env.get_current_state()
                eval_env.render() if args.render == True else None

                # Cumulative rewards:
                episode_reward = [float('{:.4f}'.format(episode_reward[agent_id]+r)) for agent_id, r in zip(range(self.args.N), r_n)]
                episode_benchmark_reward += benchmark_reward_func(error_state, eval_env.reward_min, args)
                obs_n = obs_next_n

                # Save data:
                if args.save_log:
                    eIx, eIb1 = obs_next_n[0][15:18], obs_next_n[0][-4]
                    act_list.append(action)
                    obs_list.append(np.concatenate((state_next, eIx, eIb1), axis=None))
                    cmd_list.append(np.concatenate((xd, vd, b1d, b3d, Wd), axis=None))

                # Episode termination:
                if any(done_n) or episode_timesteps == self.eval_max_steps:
                    eX = np.round(error_obs_n[0][0:3]*self.x_lim, 5) # position error [m]
                    eb1 = ang_btw_two_vectors(b1d, obs_next_n[0][6:9]) # heading error [rad]
                    print(f"eval_iter: {num_eval+1}, time_stpes: {episode_timesteps}, episode_reward: {episode_reward}, episode_benchmark_reward: {episode_benchmark_reward:.3f}, eX: {eX}, eb1: {eb1:.3f}")
                    break

            eval_reward = [eval_reward[agent_id]+epi_r for agent_id, epi_r in zip(range(self.args.N), episode_reward)]
            benchmark_reward += episode_benchmark_reward
            # Save data:
            if args.save_log:
                min_len = min(len(act_list), len(obs_list), len(cmd_list))
                log_data = np.column_stack((act_list[-min_len:], obs_list[-min_len:], cmd_list[-min_len:]))
                header = "Actions and States\n"
                header += "action[0], ..., state[0], ..., command[0], ..." 
                time_now = datetime.now().strftime("%m%d%Y_%H%M%S") 
                fpath = os.path.join('./results', 'GEOM'+'_log_'+time_now+'.dat')
                np.savetxt(fpath, log_data, header=header, fmt='%.10f') 
            sys.exit("The trained agent has been test!") if args.test_model == True else None

        # Average reward:
        eval_reward = [float('{:.4f}'.format(eval_r/self.args.num_eval)) for eval_r in eval_reward]
        benchmark_reward = float('{:.4f}'.format(benchmark_reward/self.args.num_eval))
        print("--------------------------------------------------------------------------------------------------------------------------------")
        print(f"total_timesteps: {self.total_timesteps} \t eval_reward: {eval_reward} \t benchmark_reward: {benchmark_reward}")
        print("--------------------------------------------------------------------------------------------------------------------------------")

        return eval_reward, benchmark_reward
    

    def geometricTrackingController(self, state):
        """
        Geometric controller for the UAV trajectory tracking.
        Lee, Taeyoung, Melvin Leok, and N. Harris McClamroch. "Geometric tracking control of a quadrotor UAV on SE (3)."
        https://github.com/fdcl-gwu/uav_simulator/blob/main/scripts/control.py
        """

        # States de-normalization: [-1, 1] -> [max, min]
        x, v, R, W = state_de_normalization(state, self.x_lim, self.v_lim, self.W_lim)
        
        e3 = np.array([0.,0.,1.])
        R_T = R.T
        hatW = self.hat(W)

        xd, xd_dot, xd_2dot, xd_3dot, xd_4dot, b1d, b1d_dot, b1d_2dot \
            = self.trajectory.get_desired_geometric_controller()  

        # Position control:
        # Translational error functions
        eX = x - xd     # position tracking errors 
        eV = v - xd_dot # velocity tracking errors 

        # Position integral terms
        if self.use_integral:
            self.eIX.integrate(eX + eV, self.dt) 
            self.eIX.error = np.clip(self.eIX.error, -self.sat_sigma, self.sat_sigma)
        else:
            self.eIX.set_zero()

        # Force 'f' along negative b3-axis:
        # This term equals to R.e3
        A = - self.kX@eX \
            - self.kV@eV \
            - self.m*self.g*e3 \
            + self.m*xd_2dot 
        if self.use_integral:
            A -= self.kIX@self.eIX.error

        b3 = R@e3
        b3_dot = R@hatW@e3
        f_total = -A@b3

        # Intermediate terms for rotational errors:
        ea = self.g*e3 \
            - f_total/self.m*b3 \
            - xd_2dot
        A_dot = - self.kX@eV \
                - self.kV@ea \
                + self.m*xd_3dot  

        f_dot = - A_dot@b3 \
                - A@b3_dot
        eb = - f_dot/self.m*b3 \
                - f_total/self.m*b3_dot \
                - xd_3dot
        A_2dot = - self.kX@ea \
                    - self.kV@eb \
                    + self.m*xd_4dot
        
        b3c, b3c_dot, b3c_2dot = self.deriv_unit_vector(-A, -A_dot, -A_2dot)

        hat_b1d = self.hat(b1d)
        hat_b1d_dot = self.hat(b1d_dot)
        hat_b2d_dot = self.hat(b1d_2dot)

        A2 = -hat_b1d@b3c
        A2_dot = - hat_b1d_dot@b3c - hat_b1d@b3c_dot
        A2_2dot = - hat_b2d_dot@b3c \
                    - 2.0*hat_b1d_dot@b3c_dot \
                    - hat_b1d@b3c_2dot

        b2c, b2c_dot, b2c_2dot = self.deriv_unit_vector(A2, A2_dot, A2_2dot)

        hat_b2c = self.hat(b2c)
        hat_b2c_dot = self.hat(b2c_dot)
        hat_b2c_2dot = self.hat(b2c_2dot)

        b1c = hat_b2c@b3c
        b1c_dot = hat_b2c_dot@b3c + hat_b2c@b3c_dot
        b1c_2dot = hat_b2c_2dot@b3c \
                    + 2.0*hat_b2c_dot@b3c_dot \
                    + hat_b2c@b3c_2dot

        Rd = np.vstack((b1c, b2c, b3c)).T
        Rd_dot = np.vstack((b1c_dot, b2c_dot, b3c_dot)).T
        Rd_2dot = np.vstack((b1c_2dot, b2c_2dot, b3c_2dot)).T

        Rd_T = Rd.T
        Wd = self.vee(Rd_T@Rd_dot)

        hat_Wd = self.hat(Wd)
        Wd_dot = self.vee(Rd_T@Rd_2dot - hat_Wd@hat_Wd)
        
        # Attitude control:
        RdtR = Rd_T@R
        eR = 0.5*self.vee(RdtR - RdtR.T) # attitude error vector
        eW = W - R_T@Rd@Wd # angular velocity error vector

        # Attitude integral terms:
        if self.use_integral:
            self.eIR.integrate(eR + eW, self.dt) 
            self.eIR.error = np.clip(self.eIR.error, -self.sat_sigma, self.sat_sigma)
        else:
            self.eIR.set_zero()

        M = - self.kR@eR \
            - self.kW@eW \
            + self.hat(R_T@Rd@Wd)@self.J@R_T@Rd@Wd \
            + self.J@R_T@Rd@Wd_dot
        if self.use_integral:
            M -= self.kIR@self.eIR.error
        
        # Linear scale, f_total -> [-1, 1] 
        f_norm = (f_total/4. - self.avrg_act)/self.scale_act

        return f_norm, M
        

    def hat(self, x):
        self.ensure_vector(x, 3)

        hat_x = np.array([[0.0, -x[2], x[1]], \
                          [x[2], 0.0, -x[0]], \
                          [-x[1], x[0], 0.0]])
                        
        return hat_x


    def vee(self, M):
        """Returns the vee map of a given 3x3 matrix.
        Args:
            x: (3x3 numpy array) hat of the input vector
        Returns:
            (3x1 numpy array) vee map of the input matrix
        """
        self.ensure_skew(M, 3)

        vee_M = np.array([M[2,1], M[0,2], M[1,0]])

        return vee_M


    def deriv_unit_vector(self, A, A_dot, A_2dot):
        """Returns the unit vector and it's derivatives for a given vector.
        Args:
            A: (3x1 numpy array) vector
            A_dot: (3x1 numpy array) first derivative of the vector
            A_2dot: (3x1 numpy array) second derivative of the vector
        Returns:
            q: (3x1 numpy array) unit vector of A
            q_dot: (3x1 numpy array) first derivative of q
            q_2dot: (3x1 numpy array) second derivative of q
        """

        self.ensure_vector(A, 3)
        self.ensure_vector(A_dot, 3)
        self.ensure_vector(A_2dot, 3)

        nA = np.linalg.norm(A)

        if abs(np.linalg.norm(nA)) < 1.0e-9:
            raise ZeroDivisionError('The 2-norm of A should not be zero')

        nA3 = nA * nA * nA
        nA5 = nA3 * nA * nA

        A_A_dot = A.dot(A_dot)

        q = A / nA
        q_dot = A_dot / nA \
            - A.dot(A_A_dot) / nA3

        q_2dot = A_2dot / nA \
            - A_dot.dot(2.0 * A_A_dot) / nA3 \
            - A.dot(A_dot.dot(A_dot) + A.dot(A_2dot)) / nA3 \
            + 3.0 * A.dot(A_A_dot).dot(A_A_dot)  / nA5

        return (q, q_dot, q_2dot)


    def ensure_vector(self, x, n):
        """Make sure the given input array x is a vector of size n.
        Args:
            x: (nx1 numpy array) vector
            n: (int) desired length of the vector
        Returns:
            True if the input array is satisfied with size constraint. Raises an
            exception otherwise.
        """

        np.atleast_2d(x)  # Make sure the array is atleast 2D.

        if not len(np.ravel(x)) == n:
            raise ValueError('Input array needs to be of length {}, but the size' \
                'detected is {}'.format(n, np.shape(x)))
        
        return True


    def ensure_matrix(self, x, m, n):
        """Make sure the given input array x is a matrix of size mxn.
        Args:
            x: (mxn numpy array) array
            m: (int) desired number of rows
            n: (int) desired number of columns
        Returns:
            True if the input array is satisfied with size constraint. Raises an
            exception otherwise.
        """

        np.atleast_2d(x)  # Make sure the array is atleast 2D.

        if not np.shape(x) == (m, n):
            raise ValueError('Input array needs to be of size {} x {}, but the ' \
                'size detected is {}'.format(m, n, np.shape(x)))
        
        return True


    def ensure_skew(self, x, n):
        """Make sure the given input array is a skew-symmetric matrix of size nxn.
        Args:
            x: (nxn numpy array) array
            m: (int) desired number of rows and columns
        Returns:
            True if the input array is a skew-symmetric matrix. Raises an
            exception otherwise.
        """
        self.ensure_matrix(x, n, n)
        
        if not np.allclose(x.T, -x):
            raise ValueError('Input array must be a skew-symmetric matrix')
        
        return True
        
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
    learner.eval_policy()

