import numpy as np

# Decomposing decoupled state vectors
def decoupled_obs1_decomposition(state, eIx):
    x, v, R_vec, W = state[0:3], state[3:6], state[6:15], state[15:18]
    R = R_vec.reshape(3, 3, order='F')
    b1 = R @ np.array([1.,0.,0.])
    b2 = R @ np.array([0.,1.,0.])
    b3 = R @ np.array([0.,0.,1.])
    w12 = W[0]*b1 + W[1]*b2

    return x, v, b3, w12, eIx
 

# Decomposing decoupled state vectors
def decoupled_obs2_decomposition(state, eIb1):
    R_vec, W = state[6:15], state[15:18]
    R = R_vec.reshape(3, 3, order='F')
    b1 = R @ np.array([1.,0.,0.])

    return b1, W[2], eIb1
