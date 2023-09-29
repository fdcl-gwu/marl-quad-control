import numpy as np
from numpy import linalg
from numpy.linalg import inv
from numpy.linalg import norm
from numpy.random import uniform 
from math import cos, sin, atan2, sqrt, pi
from scipy.integrate import odeint, solve_ivp
from scipy.spatial.transform import Rotation

import gymnasium as gym
from gymnasium import spaces
from gym_rotor.envs.quad import QuadEnv
from gym_rotor.envs.quad_utils import *
from gym_rotor.envs.quad_eIx_utils import *
from typing import Optional
import args_parse

class QuadEnvEIx(QuadEnv):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode: Optional[str] = None): 
        super().__init__()

        # Hyperparameters:
        parser = args_parse.create_parser()
        args = parser.parse_args()
        self.alpha = args.alpha # addressing noise or delay

        # limits of states:
        self.x_lim = 3.0 # [m]
        self.v_lim = 5.0 # [m/s]
        self.W_lim = 2*pi # [rad/s]
        self.euler_lim = 85 # [deg]
        self.eIx_lim = 10.0 
        self.low = np.concatenate([-self.x_lim * np.ones(3),  
                                   -self.v_lim * np.ones(3),
                                   -np.ones(9),
                                   -self.W_lim * np.ones(3),
                                   -self.eIx_lim * np.ones(3)])
        self.high = np.concatenate([self.x_lim * np.ones(3),  
                                    self.v_lim * np.ones(3),
                                    np.ones(9),
                                    self.W_lim * np.ones(3),
                                    self.eIx_lim * np.ones(3)])

        # Observation space:
        self.observation_space = spaces.Box(
            low=self.low, 
            high=self.high, 
            dtype=np.float64
        )


    def reset(self, env_type='train',
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        QuadEnv.reset(self, env_type)

        # Domain randomization:
        self.set_random_parameters(env_type)

        # Reset states & Normalization:
        state = np.array(np.zeros(21))
        state[6:15] = np.eye(3).reshape(1, 9, order='F')

        # Initial state error:
        self.sample_init_error(env_type)

        # x, position:
        state[0:3] = uniform(size=3,low=-self.init_x,high=self.init_x) 

        # v, velocity:
        state[3:6] = uniform(size=3,low=-self.init_v,high=self.init_v) 

        # W, angular velocity:
        state[15:18] = uniform(size=3,low=-self.init_W,high=self.init_W) 

        # R, attitude:
        roll_pitch = uniform(size=2,low=-self.init_R,high=self.init_R)
        yaw   = uniform(size=1,low=-pi, high=pi) 
        euler = np.concatenate((roll_pitch, yaw), axis=None)
        R = Rotation.from_euler('xyz', euler, degrees=False).as_matrix()
        # Re-orthonormalize:
        if not isRotationMatrix(R):
            U, s, VT = psvd(R)
            R = U @ VT.T
        R_vec = R.reshape(9, 1, order='F').flatten()
        
        # eIx, position integral error:
        state[18:21] = np.zeros(3) 

        # Normalization: [max, min] -> [-1, 1]
        x_norm, v_norm, _, W_norm, eIx_norm = eIx_state_normalization(state, self.x_lim, self.v_lim, self.W_lim, self.eIx_lim)
        self.state = np.concatenate((x_norm, v_norm, R_vec, W_norm, eIx_norm), axis=0)

        # Reset forces & moments:
        self.f  = self.m * self.g
        self.f1 = self.hover_force
        self.f2 = self.hover_force
        self.f3 = self.hover_force
        self.f4 = self.hover_force
        self.M  = np.zeros(3)
        self.fM = np.zeros((4, 1)) # Force-moment vector

        # Integral terms:
        self.eIX.set_zero() # Set all integrals to zero
        self.eIR.set_zero()

        return [np.array(self.state)]


    def observation_wrapper(self, state):
        # De-normalization: [-1, 1] -> [max, min]
        x, v, R, W, eIx = eIx_state_de_normalization(state, self.x_lim, self.v_lim, self.W_lim, self.eIx_lim)
        R_vec = R.reshape(9, 1, order='F').flatten()
        state = np.concatenate((x, v, R_vec, W, eIx), axis=0)

        # Solve ODEs: method = 'DOP853', 'BDF', 'Radau', 'RK45', ...
        sol = solve_ivp(self.EoM, [0, self.dt], state, method='DOP853')
        self.state = sol.y[:,-1]

        # Normalization: [max, min] -> [-1, 1]
        x_norm, v_norm, R, W_norm, eIx_norm = eIx_state_normalization(self.state, self.x_lim, self.v_lim, self.W_lim, self.eIx_lim)
        R_vec = R.reshape(9, 1, order='F').flatten()
        self.state = np.concatenate((x_norm, v_norm, R_vec, W_norm, eIx_norm), axis=0)

        return [self.state]
    

    def reward_wrapper(self, obs):
        # Decomposing state vectors
        x, v, R, W, eIx = eIx_state_decomposition(obs[0])

        # Errors:
        eX = x - self.xd     # position error
        eV = v - self.xd_dot # velocity error
        # Heading errors:
        eR = ang_btw_two_vectors(get_current_b1(R), self.b1d) # [rad]
        self.eIR.integrate(eR, self.dt) # b1 integral error
        self.eIR.error = clip(self.eIR.error, -self.sat_sigma, self.sat_sigma) # TODO: eb1c as state in EoM?

        # Reward function:
        reward_eX  = -self.Cx*(norm(eX, 2)**2) 
        reward_eIX = -self.CIx*(norm(eIx, 2)**2)
        reward_eR  = -self.CR*(eR/pi) # [0., pi] -> [0., 1.0]
        reward_eIR = -self.CIR*abs(self.eIR.error)
        reward_eV  = -self.Cv*(norm(eV, 2)**2)
        reward_eW  = -self.CW*(norm(W, 2)**2)

        reward = reward_eX + reward_eIX + reward_eR + reward_eIR + reward_eV + reward_eW
        return [reward]

    # https://youtu.be/iS5JFuopQsA
    def EoM(self, t, state):
        # Decomposing state vectors
        x, v, R, W, eIx = eIx_state_decomposition(state)

        # Equations of motion of the quadrotor UAV
        x_dot = v
        v_dot = self.g*self.e3 - self.f*R@self.e3/self.m
        R_vec_dot = (R@hat(W)).reshape(1, 9, order='F')
        W_dot = inv(self.J)@(-hat(W)@self.J@W + self.M)
        eIx_dot = -self.alpha*eIx + x
        state_dot = np.concatenate([x_dot.flatten(), 
                                    v_dot.flatten(),                                                                          
                                    R_vec_dot.flatten(),
                                    W_dot.flatten(),
                                    eIx_dot.flatten()])

        return np.array(state_dot)
    

    def done_wrapper(self, obs):
        # Decomposing state vectors
        x, v, _, W, _ = eIx_state_decomposition(obs[0])

        # Convert rotation matrix to Euler angles:
        # euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)

        done = False
        done = bool(
               (abs(x) >= 1.0).any() # [m]
            or (abs(v) >= 1.0).any() # [m/s]
            or (abs(W) >= 1.0).any() # [rad/s]
            # or abs(euler[0]) >= self.euler_lim # phi
            # or abs(euler[1]) >= self.euler_lim # theta
        )

        return [done]
    

    def set_random_parameters(self, env_type='train'):
        # Nominal quadrotor parameters:
        self.m = 1.994 # mass of quad, [kg]
        self.d = 0.23 # arm length, [m]
        J1, J2, J3 = 0.022, 0.022, 0.035
        self.J = np.diag([J1, J2, J3]) # inertia matrix of quad, [kg m2]
        self.c_tf = 0.0135 # torque-to-thrust coefficients
        self.c_tw = 2.2 # thrust-to-weight coefficients

        if env_type == 'train':
            uncertainty_range = 0.05 # *100 = [%]
            # Quadrotor parameters:
            m_range = self.m * uncertainty_range
            d_range = self.d * uncertainty_range
            J1_range = J1 * uncertainty_range
            J3_range = J3 * uncertainty_range
            c_tf_range = self.c_tf * uncertainty_range
            c_tw_range = self.c_tw * uncertainty_range

            self.m = uniform(low=(self.m - m_range), high=(self.m + m_range)) # [kg]
            self.d = uniform(low=(self.d - d_range), high=(self.d + d_range)) # [m]
            J1 = uniform(low=(J1 - J1_range), high=(J1 + J1_range))
            J2 = J1 
            J3 = uniform(low=(J3 - J3_range), high=(J3 + J3_range))
            self.J  = np.diag([J1, J2, J3]) # [kg m2]
            self.c_tf = uniform(low=(self.c_tf - c_tf_range), high=(self.c_tf + c_tf_range))
            self.c_tw = uniform(low=(self.c_tw - c_tw_range), high=(self.c_tw + c_tw_range))
            
            # TODO: Motor and Sensor noise: thrust_noise_ratio, sigma, cutoff_freq
            
        # Force and Moment:
        self.f = self.m * self.g # magnitude of total thrust to overcome  
                                 # gravity and mass (No air resistance), [N]
        self.hover_force = self.m * self.g / 4.0 # thrust magnitude of each motor, [N]
        self.min_force = 0.5 # minimum thrust of each motor, [N]
        self.max_force = self.c_tw * self.hover_force # maximum thrust of each motor, [N]
        self.fM = np.zeros((4, 1)) # Force-moment vector
        self.forces_to_fM = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [0.0, -self.d, 0.0, self.d],
            [self.d, 0.0, -self.d, 0.0],
            [-self.c_tf, self.c_tf, -self.c_tf, self.c_tf]
        ]) # Conversion matrix of forces to force-moment 
        self.fM_to_forces = np.linalg.inv(self.forces_to_fM)
        self.avrg_act = (self.min_force+self.max_force)/2.0 
        self.scale_act = self.max_force-self.avrg_act # actor scaling

        # print('m:',f'{self.m:.3f}','d:',f'{self.d:.3f}','J:',f'{J1:.4f}',f'{J3:.4f}','c_tf:',f'{self.c_tf:.4f}','c_tw:',f'{self.c_tw:.3f}')