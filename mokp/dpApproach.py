import numpy as np

from typing import Callable
from tqdm import tqdm
from itertools import accumulate

def paretoDominate(x: tuple, y: tuple) -> bool:
    """Determine whether x Pareto-dominates y or not."""
    return x != y and all(xi >= yi for xi, yi in zip(x, y))

def lorenzVector(x: tuple) -> tuple:
    """Return the Lorenz vector of x."""
    return tuple(accumulate(sorted(x)))

def lorenzDominate(x: tuple, y: tuple) -> bool:
    """Determine whether x Lorenz-dominates y or not."""
    return paretoDominate(lorenzVector(x), lorenzVector(y))

def ND(points: set[tuple], dominate: Callable[[tuple, tuple], bool]) -> set[tuple]:
    """Return all non-dominated points."""
    ndPoints = set()
    for x in points:
        isNonDominated = True
        for y in points:
            if x != y and dominate(y, x):
                isNonDominated = False
                break
        if isNonDominated:
            ndPoints.add(x)
    return ndPoints

def NDMerge(X: set[tuple], Y: set[tuple], dominate: Callable[[tuple, tuple], bool]) -> set[tuple]:
    """Return all non-dominated points obtained by merging two non-dominated sets."""
    if not isinstance(X, set):
        X = set(X)
    if not isinstance(Y, set):
        Y = set(Y)
    Y = Y - X

    ndPoints = set()
    # Collect all points of X that are not domineted by any point of Y.
    for x in X:
        isNonDominated = True
        for y in Y:
            if dominate(y, x):
                isNonDominated = False
                break
        if isNonDominated:
            ndPoints.add(x)
    # Do the same thing of Y.
    for y in Y:
        isNonDominated = True
        for x in X:
            if dominate(x, y):
                isNonDominated = False
                break
        if isNonDominated:
            ndPoints.add(y)
    return ndPoints

def modp(values: np.ndarray, weights: np.ndarray, capacity: int, dominate: Callable[[tuple, tuple], bool], disable: bool=True) -> set[tuple]:
    """Multi-objective dynamic programming for the Knapsack Problem."""
    m, n = values.shape
    prevDP = np.empty(shape=(capacity + 1, ), dtype=set)
    currDP = np.empty(shape=(capacity + 1, ), dtype=set)

    prevDP[:] = { tuple(0 for _ in range(n)) }
    currDP[:] = { tuple(0 for _ in range(n)) }

    for i in tqdm(range(1, m + 1), disable=disable):
        vsi = values[i - 1]
        wi = weights[i - 1]

        for j in tqdm(range(1, capacity + 1), leave=False, disable=disable):
            if j < wi:
                currDP[j] = prevDP[j]
            else:
                points = { tuple(float(xi + vi) for xi, vi in zip(p, vsi)) for p in prevDP[j - wi] }
                currDP[j] = NDMerge(points, prevDP[j], dominate=dominate)
        
        prevDP, currDP = currDP, prevDP
    return prevDP[-1]


def dpApproach(values: np.ndarray, weights: np.ndarray, capacity: int, disable: bool=True) -> dict[str, set[tuple]]:
    """"""
    paretoNDPoints = modp(values, weights, capacity, dominate=paretoDominate, disable=disable)
    lorenzNDPoints = ND(paretoNDPoints, dominate=lorenzDominate)
    return { "pareto": paretoNDPoints, "lorenz": lorenzNDPoints }

if __name__ == "__main__":
    from dataIO import loadKPData

    df, capacity = loadKPData("../data/2KP200-TA-0.dat")

    m, n = 20, 2 # 20 objects and 2 objectives.

    df = df.head(m)
    colstrs = list(df.columns)

    weights = df["w"].to_numpy(dtype=int)
    values = df[colstrs[1 : n + 1]].to_numpy(dtype=int)
    capacity = int(np.floor(weights.sum() / 2))

    results = dpApproach(values, weights, capacity, disable=False)
    print(results["pareto"])
    print(results["lorenz"])
