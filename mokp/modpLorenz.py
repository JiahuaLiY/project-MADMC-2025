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
    """Compute the set of non-dominated points from a set of points."""
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

def modp(values: np.ndarray, weights: np.ndarray, capacity: int, dominate: Callable[[tuple, tuple], bool]) -> set[tuple]:
    """Multi-objective dynamic programming for the Knapsack Problem."""
    m, n = values.shape
    prevDP = np.empty(shape=(capacity + 1, ), dtype=set)
    currDP = np.empty(shape=(capacity + 1, ), dtype=set)

    prevDP[:] = { tuple(0 for _ in range(n)) }
    currDP[:] = { tuple(0 for _ in range(n)) }

    for i in tqdm(range(1, m + 1)):
        vsi = values[i - 1]
        wi = weights[i - 1]

        for j in tqdm(range(1, capacity + 1), leave=False):
            if j < wi:
                currDP[j] = prevDP[j]
            else:
                points = {
                    tuple(int(xi + vi) for xi, vi in zip(p, vsi)) for p in prevDP[j - wi]
                }
                points |= prevDP[j]
                currDP[j] = ND(points, dominate=dominate)
        
        prevDP, currDP = currDP, prevDP
    return prevDP[-1]


def modpLorenz(values: np.ndarray, weights: np.ndarray, capacity: int) -> set[tuple]:
    paretoNDPoints = modp(values, weights, capacity, dominate=paretoDominate)
    return ND(paretoNDPoints, dominate=lorenzDominate)

if __name__ == "__main__":
    from utils import readKPData

    df, capacity = readKPData("../data/2KP200-TA-0.dat")

    m, n = 20, 2 # 20 objects and 2 objectives.

    df = df.head(m)
    colstrs = list(df.columns)

    weights = df["w"].to_numpy(dtype=int)
    values = df[colstrs[1 : n + 1]].to_numpy(dtype=int)
    capacity = int(np.floor(weights.sum() / 2))

    paretoNDPoints = modp(values, weights, capacity, dominate=paretoDominate)
    
    print(paretoNDPoints)
    print(list(map(lorenzVector, paretoNDPoints)))
    print(ND(paretoNDPoints, dominate=lorenzDominate))

    print(modpLorenz(values, weights, capacity))




    # df, _ = readKPData("../data/2KP200-TA-0.dat")

    # m, n = 100, 2

    # df = df.head(m)
    # colstrs = list(df.columns)

    # weights = df["w"].to_numpy(dtype=int)
    # values = df[colstrs[1 : n + 1]].to_numpy(dtype=int)
    # capacity = 26856

    # paretoNDPoints = modp(values, weights, capacity, dominate=paretoDominate)
    # print(paretoNDPoints)
    # [output]: {(39678, 39752), (35953, 41416), (40218, 39336), (41428, 38220), (34311, 41636), (39148, 40111), (34988, 41588), (42501, 35878), (41459, 38128), (35547, 41515), (40914, 38749), (38785, 40379), (42206, 37095), (41722, 37821), (41857, 37637), (36325, 41351), (41580, 37993), (39008, 40182), (41651, 37879), (35832, 41448), (39198, 40089), (40936, 38656), (36614, 41273), (41522, 38079), (37652, 41042), (38444, 40591), (41264, 38412), (39877, 39651), (37702, 40890), (42696, 35441), (42123, 37248), (42298, 36554), (39774, 39729), (38669, 40446), (39113, 40139), (40002, 39483), (40635, 39093), (38058, 40850), (40109, 39447), (40918, 38716), (41981, 37459), (42365, 36536), (34417, 41634), (42795, 34365), (33416, 41743), (36744, 41221), (39652, 39820), (42538, 35807), (40723, 39000), (41623, 37945), (40311, 39257), (34652, 41622), (38844, 40311), (39658, 39796), (40144, 39345), (35991, 41385), (38226, 40694), (36493, 41305), (34753, 41600), (40459, 39139), (42137, 37136), (39521, 39884), (38257, 40602), (42436, 36301), (41817, 37659), (39485, 39992), (40887, 38808), (42455, 35960), (41036, 38641), (42255, 36838), (42490, 35957), (38937, 40272), (36998, 41184), (40737, 38887), (33181, 41755), (41150, 38574), (39835, 39655), (42864, 34214), (37690, 41011), (37417, 41054), (37740, 40859), (42268, 36822), (42440, 36022), (42750, 34851), (39567, 39867), (41694, 37831), (42785, 34680), (41357, 38334), (40652, 39058), (39952, 39513), (38191, 40733), (40843, 38854), (41388, 38242), (40428, 39231), (41914, 37598), (40019, 39450), (41786, 37751), (38416, 40601), (38876, 40308), (39226, 40079), (42135, 37209), (39261, 40027), (42581, 35663), (37100, 41177), (42290, 36667), (39959, 39485), (41221, 38460), (42446, 36020), (38992, 40241), (40516, 39138), (38739, 40396), (33920, 41691), (39706, 39742), (35668, 41483), (42220, 36983), (41940, 37484), (39250, 40038), (42625, 35555), (42052, 37362), (38660, 40563), (35156, 41570), (39314, 40022), (36979, 41209), (36446, 41319), (39931, 39597)}
    # [true results] = {(39678, 39752), (35953, 41416), (40218, 39336), (41428, 38220), (34311, 41636), (39148, 40111), (34988, 41588), (42501, 35878), (41459, 38128), (40914, 38749), (35547, 41515), (38785, 40379), (42206, 37095), (41722, 37821), (41857, 37637), (36325, 41351), (41580, 37993), (39008, 40182), (41651, 37879), (35832, 41448), (39198, 40089), (40936, 38656), (41522, 38079), (37652, 41042), (36614, 41273), (38444, 40591), (41264, 38412), (39877, 39651), (37702, 40890), (42696, 35441), (42298, 36554), (42123, 37248), (39774, 39729), (38669, 40446), (39113, 40139), (40002, 39483), (40635, 39093), (40109, 39447), (38058, 40850), (40918, 38716), (41981, 37459), (42365, 36536), (34417, 41634), (42795, 34365), (36744, 41221), (33416, 41743), (39652, 39820), (42538, 35807), (41623, 37945), (40723, 39000), (40311, 39257), (34652, 41622), (38844, 40311), (39658, 39796), (40144, 39345), (35991, 41385), (38226, 40694), (36493, 41305), (34753, 41600), (42137, 37136), (39521, 39884), (40459, 39139), (38257, 40602), (42436, 36301), (41817, 37659), (39485, 39992), (40887, 38808), (42455, 35960), (41036, 38641), (42490, 35957), (42255, 36838), (38937, 40272), (36998, 41184), (40737, 38887), (41150, 38574), (33181, 41755), (39835, 39655), (42864, 34214), (37690, 41011), (37417, 41054), (37740, 40859), (42268, 36822), (42440, 36022), (42750, 34851), (39567, 39867), (41694, 37831), (42785, 34680), (40652, 39058), (41357, 38334), (36446, 41319), (39952, 39513), (38191, 40733), (40843, 38854), (41388, 38242), (41914, 37598), (40428, 39231), (40019, 39450), (41786, 37751), (38416, 40601), (38876, 40308), (39226, 40079), (42135, 37209), (39261, 40027), (42581, 35663), (37100, 41177), (42290, 36667), (41221, 38460), (42446, 36020), (38992, 40241), (40516, 39138), (38739, 40396), (33920, 41691), (39706, 39742), (35668, 41483), (42220, 36983), (41940, 37484), (39250, 40038), (42625, 35555), (42052, 37362), (38660, 40563), (35156, 41570), (39314, 40022), (36979, 41209), (39959, 39485), (39931, 39597)}
