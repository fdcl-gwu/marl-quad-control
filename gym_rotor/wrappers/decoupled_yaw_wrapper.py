import numpy as np
from numpy.linalg import inv
from numpy.random import uniform 
from scipy.integrate import odeint, solve_ivp

from gym_rotor.envs.quad import QuadEnv
from gym_rotor.envs.quad_eIx import QuadEnvEIx
from gym_rotor.envs.quad_utils import *
from gym_rotor.envs.quad_eIx_utils import *
from typing import Optional

# class DecoupledWrapper(QuadEnv):
class DecoupledWrapper(QuadEnvEIx):
    # metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None): 
        super().__init__()
        self.b3d = np.array([0.,0.,1])
        self.M3 = 0. # [Nm]

    def reset(self, env_type='train',
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=seed)
        # QuadEnv.reset(self, env_type)
        QuadEnvEIx.reset(self, env_type)

        # Reset forces & moments:
        self.fM = np.zeros((4, 1)) # Force-moment vector
        self.M3 = 0. # [Nm]

        # Agent1's obs:
        obs_1 = np.concatenate((decoupled_obs1_decomposition(self.state)), axis=None)
        # Agent2's obs:
        obs_2 = np.concatenate((decoupled_obs2_decomposition(self.state)), axis=None)

        return [obs_1, obs_2]
        
        
    def action_wrapper(self, action):

        # Linear scale, [-1, 1] -> [min_act, max_act] 
        f_total = (
            4 * (self.scale_act * action[0] + self.avrg_act)
            ).clip(4*self.min_force, 4*self.max_force)

        self.f   = f_total # [N]
        self.tau = action[1:4] 
        self.M3  = action[4] # [Nm]
        
        return action


    def observation_wrapper(self, state):
        # De-normalization: [-1, 1] -> [max, min]
        x, v, R, W, eIx = eIx_state_de_normalization(state, self.x_lim, self.v_lim, self.W_lim, self.eIx_lim)
        R_vec = R.reshape(9, 1, order='F').flatten()
        state = np.concatenate((x, v, R_vec, W, eIx), axis=0)

        # Convert each forces to force-moment:
        self.fM[0] = self.f
        b1, b2 = R@self.e1, R@self.e2
        self.fM[1] = b1.T @ self.tau + self.J[2,2]*W[2]*W[1] # M1
        self.fM[2] = b2.T @ self.tau - self.J[2,2]*W[2]*W[0] # M2
        self.fM[3] = self.M3

        """
        # FM matrix to thrust of each motor:
        forces = (self.fM_to_forces @ self.fM
            ).clip(self.min_force, self.max_force).flatten()
        self.f1 = forces[0]
        self.f2 = forces[1]
        self.f3 = forces[2]
        self.f4 = forces[3]
        """

        # Solve ODEs: method = 'DOP853', 'BDF', 'Radau', 'RK45', ...
        sol = solve_ivp(self.decouple_EoM, [0, self.dt], state, method='DOP853')
        self.state = sol.y[:,-1]

        # Normalization: [max, min] -> [-1, 1]
        x_norm, v_norm, R, W_norm, eIx_norm = eIx_state_normalization(self.state, self.x_lim, self.v_lim, self.W_lim, self.eIx_lim)
        R_vec = R.reshape(9, 1, order='F').flatten()
        self.state = np.concatenate((x_norm, v_norm, R_vec, W_norm, eIx_norm), axis=0)

        # Agent1's obs:
        obs_1 = np.concatenate((decoupled_obs1_decomposition(self.state)), axis=None)
        # Agent2's obs:
        obs_2 = np.concatenate((decoupled_obs2_decomposition(self.state)), axis=None)

        return [obs_1, obs_2]
    

    def reward_wrapper(self, obs):
        # Agent1's obs:
        x, v, b3, w12, eIx = decoupled_obs1_decomposition(self.state)
        # Agent2's obs:
        b1, W3 = decoupled_obs2_decomposition(self.state)

        # Agent1's reward:
        eX = x - self.xd     # position error
        eV = v - self.xd_dot # velocity error
        eb3 = ang_btw_two_vectors(b3, self.b3d) # [rad]

        reward_eX   = -self.Cx*(norm(eX, 2)**2) 
        reward_eIX  = -self.CIx*(norm(eIx, 2)**2)
        reward_eV   = -self.Cv*(norm(eV, 2)**2)
        reward_eb3  = -self.Cb3*(eb3/np.pi) 
        reward_ew12 = -self.Cw12*(norm(w12, 2)**2)
        rwd_1 = reward_eX + reward_eIX+ reward_eV + reward_eb3 + reward_ew12
        
        # Agent2's reward:
        b1c = -(hat(b3) @ hat(b3)) @ self.b1d # desired b1 
        eb1 = 1 - b1@b1c # b1 error
        self.eIR.integrate(eb1, self.dt) # b1 integral error
        self.eIR.error = clip(self.eIR.error, -self.sat_sigma, self.sat_sigma) # TODO: eb1c as state in EoM?

        reward_eb1  = -self.Cb1*(eb1/np.pi)
        reward_eW3  = -self.CW3*(abs(W3)**2)
        reward_eIb1 = -self.CIR*abs(self.eIR.error)
        rwd_2 = reward_eb1 + reward_eW3 + reward_eIb1

        return [rwd_1, rwd_2]


    def done_wrapper(self, obs):
        # Decomposing state vectors
        x, v, _, W, _ = eIx_state_decomposition(self.state)

        # Agent1's terminal states:
        done_1 = False
        done_1 = bool(
               (abs(x) >= 1.0).any() # [m]
            or (abs(v) >= 1.0).any() # [m/s]
            or (abs(W[0]) >= 1.0) # [rad/s]
            or (abs(W[1]) >= 1.0) # [rad/s]
        )

        # Agent2's terminal states:
        done_2 = False
        done_2 = bool(
            (abs(W[2]) >= 1.0) # [rad/s]
        )

        return [done_1, done_2]


    def decouple_EoM(self, t, state):
        # Parameters:
        m, g, J = self.m, self.g, self.J

        # Decomposing state vectors
        x, v, R, W, eIx = eIx_state_decomposition(state)

        M = self.fM[1:4].ravel()
        # Equations of motion of the quadrotor UAV
        x_dot = v
        v_dot = g*self.e3 - self.f*R@self.e3/m
        R_vec_dot = (R@hat(W)).reshape(1, 9, order='F')
        W_dot = inv(J)@(-hat(W)@J@W + M)
        eIx_dot = -self.alpha*eIx + x
        state_dot = np.concatenate([x_dot.flatten(), 
                                    v_dot.flatten(),                                                                          
                                    R_vec_dot.flatten(),
                                    W_dot.flatten(),
                                    eIx_dot.flatten()])

        return np.array(state_dot)