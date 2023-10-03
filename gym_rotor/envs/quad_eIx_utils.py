import numpy as np
from numpy import dot
from numpy import identity
from numpy import transpose
from numpy.linalg import svd
from numpy.linalg import det
from numpy.linalg import norm
from math import cos, sin, atan2, sqrt, acos, degrees


# Decomposing full-state vectors
def eIx_state_decomposition(state):
    x, v, R_vec, W, eIx = state[0:3], state[3:6], state[6:15], state[15:18], state[18:21]
    R = R_vec.reshape(3, 3, order='F')
    # Re-orthonormalize:
    if not isRotationMatrix(R):
        U, s, VT = psvd(R)
        R = U @ VT.T
        R_vec = R.reshape(9, 1, order='F').flatten()

    return x, v, R, W, eIx


# Normalization state vectors: [max, min] -> [-1, 1]
def eIx_state_normalization(state, x_lim, v_lim, W_lim, eIx_lim):
    x_norm, v_norm, W_norm, eIx_norm = state[0:3]/x_lim, state[3:6]/v_lim, state[15:18]/W_lim, state[18:21]/eIx_lim
    R_vec = state[6:15]
    R = R_vec.reshape(3, 3, order='F')
    # Re-orthonormalize:
    if not isRotationMatrix(R):
        U, s, VT = psvd(R)
        R = U @ VT.T
        R_vec = R.reshape(9, 1, order='F').flatten()

    return x_norm, v_norm, R, W_norm, eIx_norm


# De-normalization state vectors: [-1, 1] -> [max, min]
def eIx_state_de_normalization(state, x_lim, v_lim, W_lim, eIx_lim):
    x, v, W, eIx = state[0:3]*x_lim, state[3:6]*v_lim, state[15:18]*W_lim, state[18:21]*eIx_lim
    R_vec = state[6:15]
    R = R_vec.reshape(3, 3, order='F')
    # Re-orthonormalize:
    if not isRotationMatrix(R):
        U, s, VT = psvd(R)
        R = U @ VT.T
        R_vec = R.reshape(9, 1, order='F').flatten()

    return x, v, R, W, eIx


def eulerAnglesToRotationMatrix(theta) :
    # Calculates Rotation Matrix given euler angles.
    R_x = np.array([[1,              0,               0],
                    [0,  cos(theta[0]),  -sin(theta[0])],
                    [0,  sin(theta[0]),   cos(theta[0])]])

    R_y = np.array([[ cos(theta[1]),   0,  sin(theta[1])],
                    [             0,   1,              0],
                    [-sin(theta[1]),   0,  cos(theta[1])]])

    R_z = np.array([[cos(theta[2]),  -sin(theta[2]),  0],
                    [sin(theta[2]),   cos(theta[2]),  0],
                    [            0,               0,  1]])

    R = dot(R_z, dot( R_y, R_x ))

    return R


def isRotationMatrix(R):
    # Checks if a matrix is a valid rotation matrix.
    Rt = transpose(R)
    shouldBeIdentity = dot(Rt, R)
    I = identity(3, dtype = R.dtype)
    n = norm(I - shouldBeIdentity)
    return n < 1e-6


def psvd(A):
    assert A.shape == (3,3)
    U, s, VT = svd(A)
    detU = det(U)
    detV = det(VT)
    U[:,2] = U[:,2]*detU
    VT[2,:] = VT[2,:]*detV
    s[2] = s[2]*detU*detV
    # assert norm(A-U@np.diag(s)@VT) < 1e-7
    return U, s, VT.T

# def psvd(A):
#     assert A.shape == (3,3)
#     U, s, VT = svd(A)
#     det = np.det(U @ VT)
#     U[:,2] *= np.sign(det)
#     VT[2,:] *= np.sign(det)
#     s[2] *= np.sign(det)
#     assert np.allclose(A, U @ np.diag(s) @ VT)
#     return U, s, VT.T