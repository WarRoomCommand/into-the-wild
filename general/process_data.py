import pandas as pd, numpy as np, seaborn as sns, sys, matplotlib.pyplot as plt 

data = pd.read_csv(sys.argv[1], encoding="ISO-8859-1")
is_in_range = ((data['term'].astype(int) > 1994) & (data['term'].astype(int) < 2005))
spec_data = data[is_in_range]

people = ['JPStevens', 'RBGinsburg', 'DHSouter', 'SGBreyer', 'SDOConnor', 'AMKennedy', 'WHRehnquist','AScalia', 'CThomas']

to_ising = lambda x: 2 * (x - 1.0) - 1.0
output_data = spec_data.pivot(columns="justiceName", index="caseId", values="direction")
output_data = output_data[people]
output_data = output_data.transform(to_ising)
output_data = output_data.dropna(0)

corr = output_data.corr()
output_data_t = output_data.transpose()

corr.to_csv("correlations.csv", header=False, index=False)
output_data_t.to_csv("processed_data.csv", header=False, index=False)