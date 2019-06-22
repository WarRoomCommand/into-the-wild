# Fast modules written in Cython for  MFT analysis.
import numpy as np

from copy import copy
from libc.math cimport log, exp
from libc.stdlib cimport rand, RAND_MAX

def fast_sample(float gamma, start_state, J, int num_samples = 100):
    state_list = [start_state]
    old_E = ising_log_likelihood(start_state, J)
    energies = [old_E]
    
    cdef float dE
    cdef int N = 9
    
    while len(state_list) < num_samples:
        new_state = copy(state_list[-1])
        k = np.random.choice(range(N))

        # The below line is ONLY valid if J[k][k] = 0! Importantly, this is not what the cited paper 
        # does, but we know that we should be able to throw out quadratic terms. 
        dE = 4 * new_state[k] * (J[k] @ new_state)
        accept = (exp(-dE) > float(rand() / RAND_MAX))
        if accept:
            new_state[k] *= -1
            state_list.append(new_state)
            energies.append(energies[-1] + dE)
    
    return([energies, state_list])

def ising_log_likelihood(state, J):
    return(-state@J@state)
