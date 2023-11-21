import numpy as np
from numpy import interp
from numpy.linalg import norm

def benchmark_reward_func(error_state, reward_min, args):
    ex_norm = error_state[0]
    eIx = error_state[1]
    ev_norm = error_state[2]
    eb1 = error_state[3]
    eIb1 =error_state[4]
    eW_norm = error_state[5]

    reward_eX   = -args.Cx*(norm(ex_norm, 2)**2) 
    reward_eIX  = -args.CIx*(norm(eIx, 2)**2)
    reward_eV   = -args.Cv*(norm(ev_norm, 2)**2)
    reward_eb1  = -args.Cb1*(eb1)
    reward_eIb1 = -args.CIb1*abs(eIb1**2)
    reward_eW   = -args.Cw12*(norm(eW_norm, 2)**2)

    rwd = reward_eX + reward_eIX + reward_eV + reward_eb1 + reward_eIb1 + reward_eW
    
    return interp(rwd, [reward_min, 0.], [0., 1.]) # linear interpolation [0,1]