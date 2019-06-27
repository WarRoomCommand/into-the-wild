import numpy as np
import pandas as pd
import random
from copy import copy
import math


J = pd.read_csv("roots_mft.csv", encoding="ISO-8859-1", header=None).values
seed = np.array([-1,  1, -1, -1,  1,  1,  1, -1,  1])

for k in range(0, 9):
	prob = math.exp(-2 * seed[k] * (J[k] @ seed))
	print(prob)

