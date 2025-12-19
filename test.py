from mokp import dpApproach, miplApproach
from mokp.dataIO import loadKPData

import numpy as np

df, capacity = loadKPData("./data/2KP200-TA-0.dat")

m, n = 100, 2

df = df.head(m)
colstrs = list(df.columns)

weights = df["w"].to_numpy(dtype=int)
values = df[colstrs[1 : n + 1]].to_numpy(dtype=int)
capacity = int(np.floor(weights.sum() / 2))

data1 = dpApproach(values, weights, capacity)
print(data1)
