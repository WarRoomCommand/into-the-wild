import pandas as pd
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt

J = pd.read_csv("../" + sys.argv[1] + "/mft_roots_2.csv", encoding="ISO-8859-1", header=None)
sns.heatmap(J, cmap="coolwarm", vmin=-1.2, vmax=1.2)
plt.show()

