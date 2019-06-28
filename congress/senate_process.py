import pandas as pd, numpy as np, seaborn as sns, sys, matplotlib.pyplot as plt 

data = pd.read_csv("congress_votes.csv", encoding="ISO-8859-1")
is_in_range = (data["chamber"] == "senate")
spec_data = data[is_in_range]

to_ising = lambda x: (2 * x) - 1.0
output_data = spec_data.pivot_table(index="bill_id", columns="last_name", values="agree", fill_value=0.5)
output_data = output_data.transform(to_ising)

senators = output_data.columns
thresh = 2 * len(senators) / 3
for senator in senators:
	if np.sum(abs(output_data[senator])) < thresh:
		output_data = output_data.drop([senator], axis=1)
		print("Removing delinquent: ", senator)
senators = output_data.columns

print()

output_data = output_data.sort_values(output_data.index[9], axis=1)

corr = output_data.corr()

#for sen1 in senators:
#	for sen2 in senators:
#		if sen1 in corr.columns and sen2 in corr.columns:
#			if sen1 != sen2 and corr[sen1][sen2] == 1:
#				if np.sum(abs(output_data[sen1])) <= np.sum(abs(output_data[sen2])):
#					corr = corr.drop([sen1], axis=0)
#					corr = corr.drop([sen1], axis=1)
#					output_data = output_data.drop([sen1], axis=1)
#					print("Removing sheep: ", sen1)

output_data_t = output_data.transpose()

#corr.to_csv("correlations.csv", header=False, index=False)
#output_data_t.to_csv("processed_data.csv", header=False, index=False)

sns.heatmap(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
plt.show()
