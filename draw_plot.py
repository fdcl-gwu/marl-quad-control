## Importing Libraries
import os
import gymnasium as gym
from gym_rotor.wrappers.decoupled_yaw_wrapper import DecoupledWrapper
import gym_rotor
import args_parse
import numpy as np
import matplotlib.pyplot as plt
# https://www.geeksforgeeks.org/style-plots-using-matplotlib/
# https://www.dunderdata.com/blog/view-all-available-matplotlib-styles
# https://matplotlib.org/stable/gallery/color/named_colors.html
plt.style.use("seaborn-v0_8")
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 18
fontsize = 25

# Data load and indexing:
file_name = 'log_10272023_145913'
log_date = np.loadtxt(os.path.join('./results', file_name + '.dat')) 
start_index = 3
end_index = len(log_date)
is_SAVE = False

# Pre-processing:
parser = args_parse.create_parser()
args = parser.parse_args()
if args.framework_id in ("DTDE", "CTDE"):
	env = DecoupledWrapper()
elif args.framework_id == "SARL":
    env = gym.make('Quad-v0')
t = np.arange(end_index - start_index)*env.dt # time [sec]

load_act  = log_date[:, 0:5] # automatically discards the headers
load_obs  = log_date[:, 5:23+3] 
load_cmd  = log_date[:, 23+3:] 
act = load_act[start_index-2: end_index-2]
obs = load_obs[start_index-2: end_index-2]
cmd = load_cmd[start_index-2: end_index-2]

# States
x1, x2, x3 = obs[:, 0]*env.x_lim, obs[:, 1]*env.x_lim, obs[:, 2]*env.x_lim
v1, v2, v3 = obs[:, 3]*env.v_lim, obs[:, 4]*env.v_lim, obs[:, 5]*env.v_lim
R11, R21, R31 = obs[:, 6],  obs[:, 7],  obs[:, 8] 
R12, R22, R32 = obs[:, 9],  obs[:, 10], obs[:, 11]
R13, R23, R33 = obs[:, 12], obs[:, 13], obs[:, 14]  
W1, W2, W3 = obs[:, 15]*env.W_lim, obs[:, 16]*env.W_lim, obs[:, 17]*env.W_lim
eIx1, eIx2, eIx3 = obs[:, 18]*env.eIx_lim, obs[:, 19]*env.eIx_lim, obs[:, 20]*env.eIx_lim

# Actions
if args.framework_id in ("DTDE", "CTDE"):
    fM = np.zeros((4, act.shape[0])) # Force-moment vector
    f_total = (
            4 * (env.scale_act * act[:, 0] + env.avrg_act)
            ).clip(4*env.min_force, 4*env.max_force)
    tau = act[:, 1:4]

    b1, b2 = obs[:, 6:9], obs[:, 9:12]
    fM[0] = f_total
    fM[1] = np.einsum('ij,ij->i', b1, tau) + env.J[2,2]*W3*W2 # M2
    fM[2] = np.einsum('ij,ij->i', b2, tau) - env.J[2,2]*W3*W1 # M3
    fM[3] = act[:, 4]
    
    # FM matrix to thrust of each motor:
    forces = (env.fM_to_forces @ fM).clip(env.min_force, env.max_force)
    f1, f2, f3, f4 = forces[0], forces[1], forces[2], forces[3]
elif args.framework_id == "SARL":
    act = env.scale_act * act + env.avrg_act # scaling
    f1, f2, f3, f4 = act[:, 0], act[:, 1], act[:, 2], act[:, 3]

# Commands
xd1, xd2, xd3 = cmd[:, 0]*env.x_lim, cmd[:, 1]*env.x_lim, cmd[:, 2]*env.x_lim
vd1, vd2, vd3 = cmd[:, 3]*env.v_lim, cmd[:, 4]*env.v_lim, cmd[:, 5]*env.v_lim
b1d1, b1d2, b1d3 = cmd[:, 6], cmd[:, 7], cmd[:, 8]
b3d1, b3d2, b3d3 = cmd[:, 9], cmd[:, 10], cmd[:, 11]
Wd1, Wd2, Wd3 = cmd[:, 12]*env.W_lim, cmd[:, 13]*env.W_lim, cmd[:, 14]*env.W_lim

#######################################################################
############################ Plot f and M #############################
#######################################################################
fig, axs = plt.subplots(4, figsize=(25, 12))
axs[0].plot(t, fM[0], linewidth=3)
axs[0].set_ylabel('$f$ [N]', size=fontsize)

axs[1].plot(t, fM[1], linewidth=3) 
axs[1].set_ylabel('$M_1$ [Nm]', size=fontsize)

axs[2].plot(t, fM[2], linewidth=3)
axs[2].set_ylabel('$M_2$ [Nm]', size=fontsize)

axs[3].plot(t, fM[3], linewidth=3) 
axs[3].set_ylabel('$M_3$ [Nm]', size=fontsize)
axs[3].set_xlabel('Time [s]', size=fontsize)

for i in range(4):
    axs[i].set_xlim([0., t[-1]])
    axs[i].grid(True, color='white', linestyle='-', linewidth=1.0)
    axs[i].locator_params(axis='y', nbins=4)
for label in (axs[0].get_xticklabels() + axs[0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1].get_xticklabels() + axs[1].get_yticklabels()):
    label.set_fontsize(fontsize)
for label in (axs[2].get_xticklabels() + axs[2].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[3].get_xticklabels() + axs[3].get_yticklabels()):
	label.set_fontsize(fontsize)
if is_SAVE and args.framework_id in ("DTDE", "CTDE"):
    plt.savefig(os.path.join('./results', file_name[4:]+'_fM'+'.png'), bbox_inches='tight')

#######################################################################
############################# Plot Forces #############################
#######################################################################
fig, axs = plt.subplots(4, figsize=(30, 12))
axs[0].plot(t, f1, linewidth=3)
axs[0].set_ylabel('$T_1$ [N]', size=fontsize)

axs[1].plot(t, f2, linewidth=3) 
axs[1].set_ylabel('$T_2$ [N]', size=fontsize)

axs[2].plot(t, f3, linewidth=3)
axs[2].set_ylabel('$T_3$ [N]', size=fontsize)

axs[3].plot(t, f4, linewidth=3) 
axs[3].set_ylabel('$T_4$ [N]', size=fontsize)
axs[3].set_xlabel('Time [s]', size=fontsize)

for i in range(4):
    axs[i].set_xlim([0., t[-1]])
    axs[i].set_ylim([env.min_force-0.3, env.max_force+0.3])
    axs[i].grid(True, color='white', linestyle='-', linewidth=1.0)
    axs[i].locator_params(axis='y', nbins=4)
for label in (axs[0].get_xticklabels() + axs[0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1].get_xticklabels() + axs[1].get_yticklabels()):
    label.set_fontsize(fontsize)
for label in (axs[2].get_xticklabels() + axs[2].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[3].get_xticklabels() + axs[3].get_yticklabels()):
	label.set_fontsize(fontsize)
if is_SAVE:
    plt.savefig(os.path.join('./results', file_name[4:]+'_T'+'.png'), bbox_inches='tight')

###################################################################################
############################# Plot States x, v, and W #############################
###################################################################################
fig, axs = plt.subplots(3, 3, figsize=(30, 12))

axs[0, 0].plot(t, xd1, 'tab:red', linewidth=3, label='$x_{d_1}$')
axs[0, 0].plot(t, x1, linewidth=3, label='$x_1$')
axs[0, 0].set_ylabel('$x_1$ [m]', size=fontsize)
# axs[0, 0].set_title('$x_1$')

axs[0, 1].plot(t, xd2, 'tab:red', linewidth=3, label='$x_{d_2}$')
axs[0, 1].plot(t, x2, linewidth=3, label='$x_2$')
axs[0, 1].set_ylabel('$x_2$ [m]', size=fontsize)
# axs[0, 1].set_title('$x_2$')

axs[0, 2].plot(t, xd3, 'tab:red', linewidth=3, label='$x_{d_3}$')
axs[0, 2].plot(t, x3, linewidth=3, label='$x_3$')
axs[0, 2].set_ylabel('$x_3$ [m]', size=fontsize)
# axs[0, 2].set_title('$x_3$')

axs[1, 0].plot(t, vd1, 'tab:red', linewidth=3, label='$v_{d_1}$')
axs[1, 0].plot(t, v1, linewidth=3, label='$v_1$')
axs[1, 0].set_ylabel('$v_1$ [m/s]', size=fontsize)
# axs[1, 0].set_title('$v_1$')

axs[1, 1].plot(t, vd2, 'tab:red', linewidth=3, label='$v_{d_2}$')
axs[1, 1].plot(t, v2, linewidth=3, label='$v_2$')
axs[1, 1].set_ylabel('$v_2$ [m/s]', size=fontsize)
# axs[1, 1].set_title('$v_2$')

axs[1, 2].plot(t, vd3, 'tab:red', linewidth=3, label='$v_{d_3}$')
axs[1, 2].plot(t, v3, linewidth=3, label='$v_3$')
axs[1, 2].set_ylabel('$v_3$ [m/s]', size=fontsize)
# axs[1, 2].set_title('$v_3$')

axs[2, 0].plot(t, Wd1, 'tab:red', linewidth=3, label='$\Omega_{d_1}$')
axs[2, 0].plot(t, W1, linewidth=3, label='$\Omega_1$')
axs[2, 0].set_xlabel('Time [s]', size=fontsize)
axs[2, 0].set_ylabel('$\Omega_1$ [rad/s]', size=fontsize)
# axs[2, 0].set_title('$\Omega_1$')

axs[2, 1].plot(t, Wd2, 'tab:red', linewidth=3, label='$\Omega_{d_2}$')
axs[2, 1].plot(t, W2, linewidth=3, label='$\Omega_2$')
axs[2, 1].set_xlabel('Time [s]', size=fontsize)
axs[2, 1].set_ylabel('$\Omega_2$ [rad/s]', size=fontsize)
# axs[2, 1].set_title('$\Omega_2$')

axs[2, 2].plot(t, Wd3, 'tab:red', linewidth=3, label='$\Omega_{d_3}$')
axs[2, 2].plot(t, W3, linewidth=3, label='$\Omega_3$')
axs[2, 2].set_xlabel('Time [s]', size=fontsize)
axs[2, 2].set_ylabel('$\Omega_3$ [rad/s]', size=fontsize)
# axs[2, 2].set_title('$\Omega_3$')

for i in range(3):
    for j in range(3):
        axs[i, j].set_xlim([0., t[-1]])
        axs[i, j].grid(True, color='white', linestyle='-', linewidth=1.0)
        axs[i, j].legend(ncol=1, prop={'size': fontsize}, loc='lower right')
        axs[i, j].locator_params(axis='y', nbins=4)
for label in (axs[0, 0].get_xticklabels() + axs[0, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[0, 1].get_xticklabels() + axs[0, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[0, 2].get_xticklabels() + axs[0, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 0].get_xticklabels() + axs[1, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 1].get_xticklabels() + axs[1, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 2].get_xticklabels() + axs[1, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 0].get_xticklabels() + axs[2, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 1].get_xticklabels() + axs[2, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 2].get_xticklabels() + axs[2, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
if is_SAVE:
    plt.savefig(os.path.join('./results', file_name[4:]+'_x_v_W'+'.png'), bbox_inches='tight')

#########################################################################
############################# Plot States R #############################
#########################################################################
fig, axs = plt.subplots(3, 3, figsize=(30, 12))

axs[0, 0].plot(t, b1d1, 'tab:red', linewidth=3, label='$b_{1d_1}$')
axs[0, 0].plot(t, R11, linewidth=3, label='response')
axs[0, 0].set_ylabel('$R_{11}$', size=fontsize)
axs[0, 0].grid(True, color='white', linestyle='-', linewidth=1.0)
axs[0, 0].legend(ncol=1, prop={'size': fontsize}, loc='best')
# axs[0, 0].set_yticks(np.arange(0.97, 1.0, 0.01))

axs[1, 0].plot(t, b1d2, 'tab:red', linewidth=3, label='$b_{1d_2}$')
axs[1, 0].plot(t, R21, linewidth=3)
axs[1, 0].set_ylabel('$R_{21}$', size=fontsize)
axs[1, 0].legend(ncol=1, prop={'size': fontsize}, loc='best')
# axs[1, 0].set_title('$R_{21}$')

axs[2, 0].plot(t, b1d3, 'tab:red', linewidth=3, label='$b_{1d_3}$')
axs[2, 0].plot(t, R31, linewidth=3)
axs[2, 0].set_ylabel('$R_{31}$', size=fontsize)
axs[2, 0].set_xlabel('Time [s]', size=fontsize)
axs[2, 0].legend(ncol=1, prop={'size': fontsize}, loc='best')
# axs[2, 0].set_title('$R_{31}$')
# axs[2, 0].set_yticks(np.arange(-0.05, 0.08, 0.04))

axs[0, 1].plot(t, R12, linewidth=3)
axs[0, 1].set_ylabel('$R_{12}$', size=fontsize)
# axs[0, 1].set_title('$R_{12}$')

axs[1, 1].plot(t, R22, linewidth=3)
axs[1, 1].set_ylabel('$R_{22}$', size=fontsize)
# axs[1, 1].set_title('$R_{22}$')
# axs[1, 1].set_yticks(np.arange(0.97, 1.0, 0.01))

axs[2, 1].plot(t, R32, linewidth=3)
axs[2, 1].set_ylabel('$R_{32}$', size=fontsize)
axs[2, 1].set_xlabel('Time [s]', size=fontsize)
# axs[2, 1].set_title('$R_{32}$')

axs[0, 2].plot(t, b3d1, 'tab:red', linewidth=3, label='$b_{3d_1}$')
axs[0, 2].plot(t, R13, linewidth=3)
axs[0, 2].set_ylabel('$R_{13}$', size=fontsize)
axs[0, 2].legend(ncol=1, prop={'size': fontsize}, loc='best')
# axs[0, 2].set_title('$R_{13}$')
# axs[0, 2].set_yticks(np.arange(-0.10, 0.05, 0.05))

axs[1, 2].plot(t, b3d2, 'tab:red', linewidth=3, label='$b_{3d_2}$')
axs[1, 2].plot(t, R23, linewidth=3)
axs[1, 2].set_ylabel('$R_{23}$', size=fontsize)
axs[1, 2].legend(ncol=1, prop={'size': fontsize}, loc='best')
# axs[1, 2].set_title('$R_{23}$')

axs[2, 2].plot(t, b3d3, 'tab:red', linewidth=3, label='$b_{3d_3}$')
axs[2, 2].plot(t, R33, linewidth=3)
axs[2, 2].set_ylabel('$R_{33}$', size=fontsize)
axs[2, 2].set_xlabel('Time [s]', size=fontsize)
axs[2, 2].legend(ncol=1, prop={'size': fontsize}, loc='best')
# axs[2, 2].set_title('$R_{33}$')
# axs[2, 2].set_yticks(np.arange(0.990, 1.0, 0.004))

for i in range(3):
    for j in range(3):
        axs[i, j].set_xlim([0., t[-1]])
        axs[i, j].grid(True, color='white', linestyle='-', linewidth=1.0)
        axs[i, j].locator_params(axis='y', nbins=4)
for label in (axs[0, 0].get_xticklabels() + axs[0, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[0, 1].get_xticklabels() + axs[0, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[0, 2].get_xticklabels() + axs[0, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 0].get_xticklabels() + axs[1, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 1].get_xticklabels() + axs[1, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 2].get_xticklabels() + axs[1, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 0].get_xticklabels() + axs[2, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 1].get_xticklabels() + axs[2, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 2].get_xticklabels() + axs[2, 2].get_yticklabels()):
	label.set_fontsize(fontsize)

if is_SAVE:
    plt.savefig(os.path.join('./results', file_name[4:]+'_R'+'.png'), bbox_inches='tight')

##########################################################################
######################### Plot eX, eIX and eR ############################
##########################################################################
def vee(M):
    '''Returns the vee map of a given 3x3 matrix.
    Args:
        x: (3x3 numpy array) hat of the input vector
    Returns:
        (3x1 numpy array) vee map of the input matrix
    '''
    vee_M = np.array([M[2,1], M[0,2], M[1,0]])

    return vee_M

eR = np.zeros((t.size, 3))
for i in range(t.size):
    R_vec = np.array([[obs[i, 6],   obs[i, 7],  obs[i, 8]],
                      [obs[i, 9],   obs[i, 10], obs[i, 11]],
                      [obs[i, 12],  obs[i, 13], obs[i, 14]]])
    R = R_vec.reshape(3, 3, order='F')
    Rd = np.eye(3)
    Rd_T = Rd.T
    RdtR = Rd_T@R
    eR[i] = 0.5*vee(RdtR - RdtR.T) # attitude error vector

"""
class IntegralErrorVec3:
    def __init__(self,):
        self.error = np.zeros(3)
        self.integrand = np.zeros(3)

    def integrate(self, current_integrand, dt):
        self.error += (self.integrand + current_integrand) * dt / 2.0
        self.integrand = current_integrand

    def set_zero(self):
        self.error = np.zeros(3)
        self.integrand = np.zeros(3)
	
sat_sigma = 3.
eIX = IntegralErrorVec3() # Position integral error
eIX.set_zero() # Set all integrals to zero
eIX_vec = np.zeros((t.size, 3))
for i in range(t.size):
    eX = np.array([(x1 - xd1)[i], (x2 - xd2)[i], (x3 - xd3)[i]])
    eIX.integrate(-env.alpha*eIX.error + eX, env.dt)
    eIX.error = clip(eIX.error, -sat_sigma, sat_sigma)
    eIX_vec[i] = eIX.error
"""

fig, axs = plt.subplots(3, 3, figsize=(30, 12))
axs[0, 0].plot(t, x1 - xd1, linewidth=3, label='$e_{x_1}$')
axs[0, 0].set_ylabel('$e_{x_1}$ [m]', size=fontsize)

axs[0, 1].plot(t, x2 - xd2, linewidth=3, label='$e_{x_2}$')
axs[0, 1].set_ylabel('$e_{x_2}$ [m]', size=fontsize)

axs[0, 2].plot(t, x3 - xd3, linewidth=3, label='$e_{x_3}$')
axs[0, 2].set_ylabel('$e_{x_3}$ [m]', size=fontsize)

axs[1, 0].plot(t, eIx1, linewidth=3, label='$eI_{x_1}$')
#axs[1, 0].plot(t, eIX_vec[:, 0], linewidth=3, label='$eI_{x_1}$')
axs[1, 0].set_ylabel('$eI_{x_1}$', size=fontsize)

axs[1, 1].plot(t, eIx2, linewidth=3, label='$eI_{x_2}$')
#axs[1, 1].plot(t, eIX_vec[:, 1], linewidth=3, label='$eI_{x_2}$')
axs[1, 1].set_ylabel('$eI_{x_2}$', size=fontsize)

axs[1, 2].plot(t, eIx3, linewidth=3, label='$eI_{x_3}$')
#axs[1, 2].plot(t, eIX_vec[:, 2], linewidth=3, label='$eI_{x_3}$')
axs[1, 2].set_ylabel('$eI_{x_3}$', size=fontsize)

axs[2, 0].plot(t, eR[:, 0], linewidth=3, label='$e_{R_1}$')
axs[2, 0].set_ylabel('$e_{R_1}$', size=fontsize)

axs[2, 1].plot(t, eR[:, 1], linewidth=3, label='$e_{R_2}$')
axs[2, 1].set_ylabel('$e_{R_2}$', size=fontsize)

axs[2, 2].plot(t, eR[:, 2], linewidth=3, label='$e_{R_3}$')
axs[2, 2].set_ylabel('$e_{R_3}$', size=fontsize)

for i in range(3):
    for j in range(3):
        axs[i, j].set_xlim([0., t[-1]])
        axs[i, j].grid(True, color='white', linestyle='-', linewidth=1.0)
        axs[i, j].legend(ncol=1, prop={'size': fontsize}, loc='lower right')
        axs[i, j].locator_params(axis='y', nbins=4)
for label in (axs[0, 0].get_xticklabels() + axs[0, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[0, 1].get_xticklabels() + axs[0, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[0, 2].get_xticklabels() + axs[0, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 0].get_xticklabels() + axs[1, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 1].get_xticklabels() + axs[1, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[1, 2].get_xticklabels() + axs[1, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 0].get_xticklabels() + axs[2, 0].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 1].get_xticklabels() + axs[2, 1].get_yticklabels()):
	label.set_fontsize(fontsize)
for label in (axs[2, 2].get_xticklabels() + axs[2, 2].get_yticklabels()):
	label.set_fontsize(fontsize)
if is_SAVE:
    plt.savefig(os.path.join('./results', file_name[4:]+'_eX_eIX_eR'+'.png'), bbox_inches='tight')
else:
    plt.show()