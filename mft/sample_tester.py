import fast
import numpy as np
import pandas as pd


#J = pd.read_csv("jij_correct_solution.csv", encoding="ISO-8859-1", header=None).values
J = pd.read_csv("roots_mft.csv", encoding="ISO-8859-1", header=None).values
seed = np.random.choice([-1, 1], size=9)
print(seed)

results = fast.cfast_sample(gamma=0.0, start_state=seed, J=J, num_samples=16000)
