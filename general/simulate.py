from copy import deepcopy
import numpy as np
import pandas as pd
import random
import sys

# reads in Jij matrix
J = pd.read_csv("Jij_results.csv", encoding="ISO-8859-1", header=None)

# gets frozen
frozen = list()
if len(sys.argv) > 1:
	for i in range(1, len(sys.argv)):
		frozen.append(int(sys.argv[i]))

free = list()
for i in range(0, 9):
	if i not in frozen:
		free.append(i)
free = np.array(free)


# gets energy
def get_energy(state):
	energy = 0
	length = np.shape(state)[0]
	for i in range(0, length):
		for j in range(0, length):
			energy += J[i][j] * state[i] * state[j]
	energy = (-1/2) * energy
	return energy


NUM_ITERS = 10
T = 5
results = dict()
count = 0
current_energy = 0
o_energy = 0
o_state = []

for i in range(0, NUM_ITERS):
	# generate random start state
	state = np.zeros((9,))
	for i in range(0, 9):
		state[i] = random.choice([-1, 1])
	for i in frozen:
		state[i] = 1
	#print(state)
	o_state = state

	# starting energy
	current_energy = get_energy(state)
	o_energy = current_energy

	# run until state converges 
	total_iters = 0
	while T > 0:
			
		# picks random justice
		#poss_flip = np.random.randint(low=0, high=9)
		poss_flip = np.random.choice(free)

		# sees how the energy changes
		poss_state = deepcopy(state)
		poss_state[poss_flip] *= -1
		poss_energy = get_energy(poss_state)
		#poss_energy = change_energy(current_energy, state, poss_flip)

		# determines if one should flip
		flip = True
		delta_E = poss_energy - current_energy
		if delta_E > 0:
			if np.random.random() > np.exp(-1 * delta_E / T):
				flip = False

		if flip:
			state = deepcopy(poss_state)
			current_energy = poss_energy

		if total_iters % 10 == 0:
			T -= .001

		if total_iters % 250 == 0:
			print(str(T) + "\t" + str(current_energy))

		total_iters += 1

	state_str = np.array2string(state)
	if state_str not in results:
		results[state_str] = 0
	results[state_str] += 1
	count += 1


for state in results:
	#print(str(state) + "\t" + str(results[state]*100/count))
	print(str(o_state) + " -> " + str(state))
	print(str(o_energy) + " -> " + str(current_energy))




	





