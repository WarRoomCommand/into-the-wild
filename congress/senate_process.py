import pandas as pd, numpy as np, seaborn as sns, sys, matplotlib.pyplot as plt 

data = pd.read_csv("congress_votes.csv", encoding="ISO-8859-1")
is_in_range = (data["chamber"] == "senate")
spec_data = data[is_in_range]

to_ising = lambda x: (2 * x) - 1.0
output_data = spec_data.pivot_table(index="bill_id", columns="last_name", values="agree", fill_value=0)
output_data = output_data.transform(to_ising)
#output_data = output_data.sort_values(by=["s36-2017"], axis=1)

corr = output_data.corr()
output_data_t = output_data.transpose()

corr.to_csv("correlations.csv", header=False, index=False)
output_data_t.to_csv("processed_data.csv", header=False, index=False)

sns.heatmap(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
plt.show()