# coding: utf-8
# Imports are for voting_data, C, pvec, pbin, J_diag_factor
import pandas as pd
import numpy as np
from math import *
import scipy.linalg


sup_court_data = pd.read_csv("SCDB_2018_02_justiceCentered_Citation.csv",encoding = "ISO-8859-1")

is_rehnquist = ((sup_court_data['term'].astype(int) > 1994) & (sup_court_data['term'].astype(int) < 2005))
rehnquist_data = sup_court_data[is_rehnquist]

justices = ['JPStevens',
            'RBGinsburg',
            'DHSouter',
            'SGBreyer',
            'SDOConnor',
            'AMKennedy',
            'WHRehnquist',
            'AScalia',
            'CThomas']

to_ising = lambda x: 2 * (x - 1.0) - 1.0
voting_data = rehnquist_data.pivot(columns="justiceName", index="caseId", values="direction")
voting_data = voting_data[justices]
voting_data = voting_data.transform(to_ising)
voting_data = voting_data.dropna(0)


# Determines p_i and p_{ij} vectors, denoted as pvec and pbin

pvec = np.sum((voting_data+1.0)/2) / len(voting_data)
pbin = np.zeros((9,9))
voting_data_qubo = np.array(np.transpose((voting_data + 1.0)/2))
for i in range(9):
    for j in range(9):
        bin_occ = voting_data_qubo[i] * voting_data_qubo[j]
        pbin[i][j] = np.average(bin_occ)

N = 9
C = voting_data.corr()


ceig = scipy.linalg.eig(C)
cvals, cvecs = ceig
cvals = np.real(cvals)
V = cvecs
Vinv = np.transpose(cvecs)


J_diag_factor = np.array([[(((pbin[i][j] - pvec[i]*pvec[j]) * (pvec[i] - 0.5))/(pvec[i] * (1-pvec[i])))\
                           - pvec[j] for j in range(N)] for i in range(N)])
