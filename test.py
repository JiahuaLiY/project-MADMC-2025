from mokp import dpApproach, milpApproach
from mokp.dataIO import loadKPData, saveData

import numpy as np
from tqdm import tqdm

df, capacity = loadKPData("./data/2KP200-TA-0.dat")

n = 3
indices = (df.columns)[1 : n + 1]
owaWeights = list(range(n, 0, -1))

seeds = [192, 302, 337, 875, 177]
for m in tqdm(range(10, 90, 10)):
    for seed in tqdm(seeds, leave=False):
        np.random.seed(seed)

        dfPrime = df.sample(m)
        weights = dfPrime["w"].to_numpy(dtype=int)
        values = dfPrime[indices].to_numpy(dtype=int)
        capacity = int(np.floor(weights.sum() / 2))

        data1 = dpApproach(values, weights, capacity)
        data2 = milpApproach(values, weights, capacity, owaWeights)

        saveData(f"data/dpData/{n}KP{m}-DP-SEED-{seed}.log", data1)
        saveData(f"data/milpData/{n}KP{m}-MILP-SEED-{seed}.log", data2)
