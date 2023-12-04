"""
Reference: https://github.com/fdcl-gwu/uav_simulator/blob/main/scripts/trajectory.py
"""
import datetime
import numpy as np
from gym_rotor.envs.quad_utils import *

class TrajectoryGeneration:
    def __init__(self, env):
        """----------------------------------------------------------
            self.mode == 0 or self.mode == 1:  # idle and warm-up
        -------------------------------------------------------------
            self.mode == 2:  # take-off
        -------------------------------------------------------------
            self.mode == 3:  # landing
        -------------------------------------------------------------
            self.mode == 4:  # stay (hovering)
        -------------------------------------------------------------
            self.mode == 5:  # circle
        ----------------------------------------------------------"""
        self.mode = 0
        self.is_mode_changed = False
        self.is_landed = False
        self.e1 = env.e1

        self.is_realtime = False # if False, it is sim_time
        self.t0 = datetime.datetime.now()
        self.t = 0.0
        self.t_traj = 0.0
        self.dt = env.dt

        self.x_lim, self.v_lim, self.W_lim = env.x_lim, env.v_lim, env.W_lim

        self.xd_norm, self.vd_norm, self.Wd_norm = np.zeros(3), np.zeros(3), np.zeros(3)
        self.xd, self.vd, self.Wd = np.zeros(3), np.zeros(3), np.zeros(3)
        self.b1d = np.array([1.,0.,0.]) # desired heading direction
        self.b3d = np.array([0.,0.,1.])

        self.x_norm, self.v_norm, self.W_norm = np.zeros(3), np.zeros(3), np.zeros(3)
        self.x, self.v, self.W = np.zeros(3), np.zeros(3), np.zeros(3)
        self.R = np.identity(3)

        self.x_init, self.v_init = np.zeros(3), np.zeros(3)
        self.R_init, self.W_init = np.identity(3), np.zeros(3)
        self.b1_init = np.zeros(3)
        self.theta_init = 0.0

        self.trajectory_started  = False
        self.trajectory_complete = False
        
        # Manual mode:
        self.manual_mode = False
        self.manual_mode_init = False
        self.x_offset = np.zeros(3)
        self.yaw_offset = 0.0

        # Take-off:
        self.takeoff_end_height = -1.5  # [m]
        self.takeoff_velocity = -0.3  # [m/s]

        # Landing:
        self.landing_velocity = 1.0  # [m/s]
        self.landing_motor_cutoff_height = -0.25  # [m]

        # Circle:
        self.num_circles = 2
        self.circle_linear_v = 0.4
        self.circle_W = 0.8
        self.circle_radius = 0.7


    def get_desired(self, state, mode):
        # De-normalize state: [-1, 1] -> [max, min]
        self.x_norm, self.v_norm, self.W_norm = state[0:3], state[3:6], state[15:18]
        self.x, self.v, self.R, self.W = state_de_normalization(state, self.x_lim, self.v_lim, self.W_lim)

        # Generate desired traj: 
        if mode == self.mode:
            self.is_mode_changed = False
        else:
            self.is_mode_changed = True
            self.mode = mode
            self.mark_traj_start()
        self.calculate_desired()

        # Normalization goal state vectors: [max, min] -> [-1, 1]
        self.xd_norm = self.xd/self.x_lim
        self.vd_norm = self.vd/self.v_lim
        self.Wd_norm = self.Wd/self.W_lim

        return self.xd_norm, self.vd_norm, self.b1d, self.b3d, self.Wd_norm

    
    def calculate_desired(self):
        if self.manual_mode:
            self.manual()
            return
        
        if self.mode == 0 or self.mode == 1:  # idle and warm-up
            self.set_desired_states_to_zero()
            self.mark_traj_start()
        elif self.mode == 2:  # take-off
            self.takeoff()
        elif self.mode == 3:  # land
            self.land()
        elif self.mode == 4:  # stay
            self.stay()
        elif self.mode == 5:  # circle
            self.circle()


    def mark_traj_start(self):
        self.trajectory_started  = False
        self.trajectory_complete = False

        self.manual_mode_init = False
        self.manual_mode = False
        self.is_landed = False

        self.t0 = datetime.datetime.now()
        self.t = 0.0
        self.t_traj = 0.0

        self.x_offset = np.zeros(3)
        self.yaw_offset = 0.

        self.update_initial_state()


    def mark_traj_end(self, switch_to_manual=False):
        self.trajectory_complete = True

        if switch_to_manual:
            self.manual_mode = True


    def update_initial_state(self):
        self.x_init = np.copy(self.x)
        self.v_init = np.copy(self.v)
        self.R_init = np.copy(self.R)
        self.W_init = np.copy(self.W)
        self.b1_init = self.get_current_b1()
        self.theta_init = np.arctan2(self.b1_init[1], self.b1_init[0])


    def set_desired_states_to_zero(self):
        self.xd, self.vd, self.Wd = np.zeros(3), np.zeros(3), np.zeros(3)
        self.b1d = np.array([1.,0.,0.]) # desired heading direction
        self.b3d = np.array([0.,0.,1.])

    
    def set_desired_states_to_current(self):
        self.xd = np.copy(self.x)
        self.vd = np.copy(self.v)
        self.b1d = np.array([1.,0.,0.]) #TODO: self.get_current_b1()

    
    def get_current_b1(self):
        b1 = self.R.dot(self.e1)
        theta = np.arctan2(b1[1], b1[0])
        return np.array([np.cos(theta), np.sin(theta), 0.])


    def update_current_time(self):
        if self.is_realtime == True:
            t_now = datetime.datetime.now()
            self.t = (t_now - self.t0).total_seconds()
        else:
            self.t = self.t + self.dt


    def manual(self):
        if not self.manual_mode_init:
            self.set_desired_states_to_current()
            self.update_initial_state()

            self.manual_mode_init = True
            self.x_offset = np.zeros(3)
            self.yaw_offset = 0.

            print('Switched to manual mode')
        
        self.xd = self.x_init + self.x_offset
        self.vd = np.zeros(3) # (self.xd - self.x) / 1.0

        theta = self.theta_init + self.yaw_offset
        self.b1d = np.array([1.,0.,0.]) #TODO: np.array([np.cos(theta), np.sin(theta), 0.0])


    def takeoff(self):
        if not self.trajectory_started:
            self.set_desired_states_to_zero()

            # Take-off starts from the current horizontal position:
            self.xd[0] = self.x[0]
            self.xd[1] = self.x[1]
            self.x_init = self.x

            self.t_traj = (self.takeoff_end_height - self.x[2]) / self.takeoff_velocity

            # Set the take-off yaw to the current yaw:
            self.b1d = np.array([1.,0.,0.]) #TODO: self.get_current_b1()

            self.trajectory_started = True

        self.update_current_time()

        if self.t < self.t_traj:
            self.xd[2] = self.x_init[2] + self.takeoff_velocity * self.t 
            #self.xd_2dot[2] = self.takeoff_velocity
        else:
            if self.waypoint_reached(self.xd, self.x, 0.04):
                self.xd[2] = self.takeoff_end_height
                self.vd[2] = 0.

                if not self.trajectory_complete:
                    print('Takeoff complete\nSwitching to manual mode')
                
                self.mark_traj_end(True)


    def waypoint_reached(self, waypoint, current, radius):
        delta = waypoint - current
        
        if abs(np.linalg.norm(delta) < radius):
            return True
        else:
            return False
        

    def land(self):
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.t_traj = (self.landing_motor_cutoff_height - self.x[2]) / self.landing_velocity

            # Set the take-off yaw to the current yaw:
            self.b1d = np.array([1.,0.,0.]) #TODO: self.get_current_b1()

            self.trajectory_started = True

        self.update_current_time()

        if self.t < self.t_traj:
            self.xd[2] = self.x_init[2] + self.landing_velocity * self.t
            #self.xd_2dot[2] = self.landing_velocity
        else:
            if self.x[2] > self.landing_motor_cutoff_height:
                self.xd[2] = self.landing_motor_cutoff_height
                self.vd[2] = 0.

                if not self.trajectory_complete:
                    print('Landing complete')

                self.mark_traj_end(False)
                self.is_landed = True
            else:
                self.xd[2] = self.landing_motor_cutoff_height
                self.vd[2] = self.landing_velocity

            
    def stay(self):
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.trajectory_started = True
        
        self.mark_traj_end(True)


    def circle(self):
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.trajectory_started = True

            self.circle_center = np.copy(self.x)
            self.circle_W = 2 * np.pi / 8

            self.t_traj = self.circle_radius / self.circle_linear_v \
                        + self.num_circles * 2 * np.pi / self.circle_W

        self.update_current_time()

        if self.t < self.circle_radius / self.circle_linear_v:
            self.xd[0] = self.circle_center[0] + self.circle_linear_v * self.t
            self.vd[0] = self.circle_linear_v

        elif self.t < self.t_traj:
            circle_W = self.circle_W
            circle_radius = self.circle_radius

            t = self.t - circle_radius / self.circle_linear_v
            th = circle_W * t

            # x-axis:
            self.xd[0] = circle_radius * np.cos(th) + self.circle_center[0]
            self.vd[0] = - circle_radius * circle_W * np.sin(th)

            # y-axis:
            self.xd[1] = circle_radius * np.sin(th) + self.circle_center[1]
            self.vd[1] = circle_radius * circle_W * np.cos(th)

            # yaw-axis:
            w_b1d = 0.1*np.pi
            th_b1d = w_b1d * t
            self.b1d = np.array([np.cos(th_b1d), np.sin(th_b1d), 0]) #TODO: np.array([1.,0.,0.])
        else:
            self.mark_traj_end(True)
            

    def get_error_state(self, obs_n, framework):
        # Normalized error obs:
        ex_norm = self.x_norm - self.xd_norm # position error
        ev_norm = self.v_norm - self.vd_norm # velocity error
        eW_norm = self.W_norm - self.Wd_norm # ang vel error

        # Compute yaw angle error: 
        b1 = self.R @ np.array([1.,0.,0.])
        b2 = self.R @ np.array([0.,1.,0.])
        b3 = self.R @ np.array([0.,0.,1.])
        b1c = -(hat(b3) @ hat(b3)) @ self.b1d # desired b1 
        eb1 = norm_ang_btw_two_vectors(b1c, b1) # b1 error, [-1, 1)
        if framework in ("DTDE", "CTDE"):
            # Agent1's obs:
            eIx = obs_n[0][12:15]
            ew12 = eW_norm[0]*b1 + eW_norm[1]*b2
            obs_1 = np.concatenate((ex_norm, ev_norm, b3, ew12, eIx), axis=None)
            # Agent2's obs:
            eW3_norm, eIb1 = eW_norm[2], obs_n[1][5]
            obs_2 = np.concatenate((b1, eW3_norm, eb1, eIb1), axis=None)
            error_obs_n = [obs_1, obs_2]
        elif framework == "SARL":
            # Single-agent's obs:
            eIx, eIb1 = obs_n[0][18:21], obs_n[0][22]
            R_vec = self.R.reshape(9, 1, order='F').flatten()
            obs = np.concatenate((ex_norm, ev_norm, R_vec, eW_norm, eIx, eb1, eIb1), axis=None)
            error_obs_n = [obs]
        error_state = (ex_norm, eIx, ev_norm, eb1, eIb1, eW_norm)
        
        return error_obs_n, error_state