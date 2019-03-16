import math
import numpy as np
import pandas as pd
import random

J = pd.read_csv("Jij_results.csv", encoding="ISO-8859-1", header=None)

states = dict()
for i in range(0, 256):
	state_str = str(bin(i))[2:]
	length = len(state_str)
	while length < 9:
		state_str = "0" + state_str
		length += 1
	states[state_str] = 0

def get_energy(state):
	energy = 0
	length = 9
	for i in range(0, length):
		for j in range(0, length):
			state_i = (int(state[i]) * 2) - 1
			state_j = (int(state[j]) * 2) - 1
			energy += J[i][j] * state_i * state_j
	energy = (-1/2) * energy
	return energy

for i in states:
	states[i] = math.exp(-1 * get_energy(i))

sum_probs = 0
for i in states:
	sum_probs += states[i]

for i in states:
	states[i] = states[i] * 100/ sum_probs
	print(str(i) + "\t" + str(states[i]))


