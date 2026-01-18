# project-MADMC-2025
Dans ce projet, nous avons implémenté deux approches pour générer l'ensemble des points non dominés au sens de Lorenz pour le problème du sac à dos multi-objectifs.  
La source principale utilisée pour réaliser nos comparaisons est **["2KP200-TA-0.dat"](./data/2KP200-TA-0.dat)**, qui contient les données d'une instance du problème du sac à dos multi-objectifs à 200 objets et 6 critères. Nous n'avons utilisé que des sous-ensembles de données de cette instance pour réaliser nos comparaisons.

## Prérequis
* **[numpy](https://numpy.org)**
* **[pandas](https://pandas.pydata.org/)**
* **[gurobipy](https://pypi.org/project/gurobipy/)**
* **[SWIG](https://www.swig.org/)**

Pour installer les trois premières dépendances, veuillez utiliser la commande:
```bash
pip install numpy pandas gurobipy
```

## Approche en deux phases
### Description
Dans cette première approche, la génération de points non dominés au sens de Lorenz se divise en deux phases. Tout d'abord, elle génère tous les points non dominés au sens de Pareto pour le problème du sac à dos multi-objectifs en utilisant la programmation dynamique multi-objectif, c'est pourquoi nous appelons cette approche: ```dpApproach``` dans la suite du projet. Ensuite, elle filtre les points obtenus dans la première phase, en ne conservant que ceux qui sont non dominés au sens de Lorenz.

### Usage
```python
import numpy as np
values = np.array([[4, 6, 3],
                   [3, 7, 9],
                   [9, 6, 1],
                   [6, 3, 9],
                   [6, 7, 4]])
weights = np.array([7, 5, 7, 3, 7])
capacity = 14

from mokp import dpApproach
data = dpApproach(values, weights, capacity)
print(data)
# [out]
# {
#     'pareto': {(9, 14, 13), (12, 10, 13), (15, 13, 5), (12, 13, 10), (9, 10, 18), (15, 9, 10)},
#     'lorenz': {(9, 14, 13), (12, 10, 13), (9, 10, 18), (12, 13, 10)},
#     'number-of-objectives': 3,
#     'number-of-items': 5,
#     'runtime': 4.601478576660156e-05
# }
```
**Attention**: bien que la plupart du code soit implémenté en Python, la première phase de la fonction ```dpApproach``` est implémenté en C et interagit avec le code via un **wrapper** généré par SWIG. Par conséquent, avant d'utiliser la fonction ```dpApproach```, veuillez d'abord exécuter le fichier ```compile.sh``` pour générer ce wrapper.

## Approche MILP
### Description
Dans cette deuxième approche, la génération est réalisée par une procédure itérative en résolvant des programmes linéaires à variables mixtes, c'est pourquoi nous appelons également cette approche: ```milpApproach``` dans la suite du projet. Cette approche est principalement fondée sur la formulation linéaire du modèle OWA (Ordered Weighted Averaging). Pour plus de détails, nous vous invitons à consulter les papiers [[1]](./docs/LorenzDominance.pdf) et [[2]](./docs/SylvaCrema.pdf).

### Usage
```python
from mokp import milpApproach

# Any strictly descrasing weight vector.
owaWeights = [3, 2, 1]

data = milpApproach(values, weights, capacity, owaWeights)
print(data)
# [out]
# {
#     'pareto': {(9, 14, 13), (9, 10, 18), (12, 13, 10)},
#     'lorenz': {(9, 14, 13), (9, 10, 18), (12, 13, 10)},
#     'number-of-objectives': 3,
#     'number-of-items': 5,
#     'runtime': 0.008540868759155273
# }
# We note that the simplied version of the milpApproach does
# not generate all Lorenz non-dominated points, since it
# does not distinguish between two vectors whose Lorenz vector
# is identical (e.g., (12, 10, 13) and (12, 13, 10) ), as is
# the case above.

# We have implemented the improved version that generates
# all points. Simply set the flag: findAllLorenzND to true
# to use it.
data = milpApproach(values, weights, capacity, owaWeights, findAllLorenzND=True)
print(data)
# [out]
# {
#     'pareto': {(9, 14, 13), (12, 10, 13), (9, 10, 18), (12, 13, 10)},
#     'lorenz': {(9, 14, 13), (12, 10, 13), (9, 10, 18), (12, 13, 10)},
#     'number-of-objectives': 3,
#     'number-of-items': 5,
#     'runtime': 0.025630950927734375
# }
```

## Usage pour les tests et les comparaisons
```python
import numpy as np
from mokp import dpApproach, milpApproach
from mokp.dataIO import loadKPData, saveData, loadData

#################### Create an instance ####################
n, m = 2, 20 # 2 objectives and 20 items.
df, capacity = loadKPData("./data/2KP200-TA-0.dat")

# Take n first objectives.
indices = list(df.columns)[1 : n + 1] # ["v1", "v2", ..., "vn"]

# Take m items.
dfPrime = df.head(m) # m first items.
# dfPrime = df.tail(m) # m last items.
# dfPrime = df.sample(m) # m random items.

# Extract data.
values = dfPrime[indices].to_numpy(dtype=int)
weights = dfPrime["w"].to_numpy(dtype=int)
capacity = int(np.floor(weights.sum() / 2))


#################### Run tests ####################
dpData = dpApproach(values, weights, capacity)

owaWeights = list(range(n, 0, -1)) # [n, n - 1, ..., 1]
milpData = milpApproach(values, weights, capacity, owaWeights)


#################### Save tests data ####################
saveData(f"data/dpData/{n}KP{m}-DP-test.log", dpData)
saveData(f"data/milpData/{n}KP{m}-MILP-test.log", milpData)


#################### load tests data ####################
# dpData = loadData(f"data/dpData/{n}KP{m}-DP-test.log")
# milpData = loadData(f"data/milpData/{n}KP{m}-MILP-test.log")
```

## Références
[[1] Mohammed Bederina, Djamal Chaabane, and Thibaut Lust. Optimizing a linear function
over the lorenz-efficient set of multi-objective combinatorial optimization problems. Technical
report, 2025.](./docs/LorenzDominance.pdf)

[[2] John Sylva and Alejandro Crema. A method for finding the set of non-dominated vectors
for multiple objective integer linear programs. European Journal of Operational Research,
158(1) :46–55, 2004.](./docs/SylvaCrema.pdf)
