# Fast modules written in Cython for  MFT analysis.
import numpy as np

from copy import copy
from libc.math cimport log, exp
from libc.stdlib cimport rand, RAND_MAX

cdef extern from "stdlib.h":
    double drand48()
    
def error(msg):
    with open("error.txt", "a+") as f:
        f.write(msg + '\n')
def clear_file():
    with open("error.txt", "w+") as f:
        f.close()
       
    
def fib():
    a, b = 1, 1
    while True:
        yield(a)
        a, b = b, a+b

def cfast_sample(float gamma, start_state, J, int num_samples = 100):
    clear_file()
    error("start run")
    state_list = [start_state]
    old_E = ising_log_likelihood(start_state, J)
    
    energies = np.zeros(num_samples)
    probs = np.zeros(num_samples)
    
    #energies[0] = old_E
    probs[0] = exp(-old_E)
    
    cdef float dE
    cdef int N = 9
    
    index = 1
    error("enter loop")
    while len(state_list) < num_samples:
        error("\t Loop point 1")
        new_state = copy(state_list[-1])
        error("\t Loop point 2")
        k = int(rand() % N)
    
        # The below line is ONLY valid if J[k][k] = 0! Importantly, this is not what the cited paper 
        # does, but we know that we should be able to throw out quadratic terms. 
        dE = 2 * new_state[k] * (J[k] @ new_state)
        rel_prob = exp(-dE)
        accept = (rel_prob > np.random.rand())
        if accept:
            new_state[k] *= -1
            state_list.append(new_state)
            #energies[index] = energies[index-1] + dE
            probs[index] = rel_prob * probs[index - 1]
            index += 1
    # NOTE: Not including the regularization term because it's not state dependent, so a Markov
    # chain doesn't actually need that information. 
    return([probs / np.sum(probs), np.array(state_list)])


def my_test_sample(float gamma, start_state, J, int num_samples = 100):
    clear_file()
    state_list = [start_state]
    old_E = ising_log_likelihood(start_state, J)
    
    energies = np.zeros(num_samples)
    probs = np.zeros(num_samples)
    
    #energies[0] = old_E
    probs[0] = exp(-old_E)
    
    cdef float dE
    cdef int N = 9
    
    index = 1
    num_attempts = 0
    
    # Burn in 
    for i in range(200):
        new_state = copy(state_list[-1])
        k = int(rand() % N)
    
        # The below line is ONLY valid if J[k][k] = 0! Importantly, this is not what the cited paper 
        # does, but we know that we should be able to throw out quadratic terms. 
        dE = 2 * new_state[k] * (J[k] @ new_state)
        rel_prob = exp(-dE)
        accept = (rel_prob > float(rand()) / RAND_MAX)
        if accept:
            state_list[0][k] *= -1
            probs[0] *= rel_prob
            
    # Sampling 
    T = 5
    while len(state_list) < num_samples:
        new_state = copy(state_list[-1])
        k = int(rand() % N)
        
        # The below line is ONLY valid if J[k][k] = 0! Importantly, this is not what the cited paper 
        # does, but we know that we should be able to throw out quadratic terms. 
        dE = 2 * new_state[k] * (J[k] @ new_state)
        rel_prob = exp(-dE / T)
        accept = (rel_prob > np.random.rand())
        num_attempts += 1
        if accept:
            new_state[k] *= -1
            state_list.append(new_state)
            probs[index] = rel_prob * probs[index - 1]
            index += 1
            num_attempts = 0
    # NOTE: Not including the regularization term because it's not state dependent, so a Markov
    # chain doesn't actually need that information. 
    return([probs / np.sum(probs), np.array(state_list)])


def ising_log_likelihood(state, J):
    return(-0.5 * state@J@state)
