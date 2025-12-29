import numpy as np
import gurobipy as gp

from time import time

def initOWAModel(values: np.ndarray, weights: np.ndarray, capacity: int, owaWeights: np.ndarray) -> tuple[gp.Model, tuple[gp.tupledict[int, gp.Var]]]:
    """Initialize the basic OWA model for the knapsack problem with the given parameters."""
    m, n = values.shape

    model = gp.Model(env = gp.Env(params={"LogToConsole": 0}))

    xs = model.addVars(m, vtype=gp.GRB.BINARY)
    fs = model.addVars(n, vtype=gp.GRB.CONTINUOUS)
    rs = model.addVars(n, lb=float("-inf"), vtype=gp.GRB.CONTINUOUS)
    bs = model.addVars(n, n, vtype=gp.GRB.CONTINUOUS)
    model.update()

    objfunc = gp.LinExpr()
    for k in range(n - 1):
        objfunc += (owaWeights[k] - owaWeights[k + 1]) * ((k + 1) * rs[k] - gp.quicksum(bs[k, i] for i in range(n)))
    objfunc += owaWeights[n - 1] * (n * rs[n - 1] - gp.quicksum(bs[n - 1, i] for i in range(n)))
    model.setObjective(objfunc, gp.GRB.MAXIMIZE)

    model.addConstr(gp.quicksum(weights[j] * xs[j] for j in range(m)) <= capacity)
    model.addConstrs((fs[i] == gp.quicksum(values[j, i] * xs[j] for j in range(m)) for i in range(n)))
    model.addConstrs(((rs[k] - bs[k, i] <= fs[i]) for i in range(n) for k in range(n)))

    return model, (xs, fs, rs, bs)

def findSimilar(y: tuple, lorenzVec: tuple, values: np.ndarray, weights: np.ndarray, capacity: int, owaWeights: np.ndarray) -> set[tuple]:
    """Find all non-dominated points with same lorenz vector as y."""
    similarNDPoints = {y}
    _, n = values.shape

    model, (_, fs, rs, bs) = initOWAModel(values, weights, capacity, owaWeights)
    zs = [model.addVars(n, vtype=gp.GRB.BINARY)]
    model.update()

    model.addConstrs((((k + 1) * rs[k] - gp.quicksum(bs[k, i] for i in range(n)) == lorenzVec[k]) for k in range(n)))
    model.addConstrs(((fs[i] >= (y[i] + 1) * zs[-1][i]) for i in range(n)))
    model.addConstr(gp.quicksum(zs[-1][k] for k in range(n)) >= 1)

    model.optimize()
    while model.status == gp.GRB.OPTIMAL:
        y = tuple(round(fs[k].X) for k in range(n))
        similarNDPoints.add(y)

        zs.append(model.addVars(n, vtype=gp.GRB.BINARY))
        model.update()

        model.addConstrs(((fs[i] >= (y[i] + 1) * zs[-1][i]) for i in range(n)))
        model.addConstr(gp.quicksum(zs[-1][k] for k in range(n)) >= 1)

        model.optimize()
    
    return similarNDPoints

def milpApproach(values: np.ndarray,
                 weights: np.ndarray,
                 capacity: int,
                 owaWeights: np.ndarray,
                 findAllLorenzND: bool=False,
                 verbose: bool=False) -> dict:
    """Mixed Integer Linear Programming based method to generate Lorenz non-dominated points
    for the knapsack problem.
    
    Parameters:
        values (```np.ndarray```): 2D array representing the valuation of each item
            in each objective function.

        weights (```np.ndarray```): 1D array representing the weight of each item.

        capacity (```int```): Maximum capacity.

        owaWeights (```int```): List of OWA weights.

        findAllLorenzND (```bool```): If true, then apply the adaptation method to find
            all Lorenz non-dominated points; otherwise, return only the Lorenz non-dominated
            points with distinct Lorenz vectors.

        verbose (```bool```): Display detailed execution.

    Returns:
        out (```dict```): A dictionary containing all Lorenz non-dominated points generated
            with some additional data, such as runtime.
    """
    lorenzNDPoints = set()
    m, n = values.shape
    it = 1

    if verbose:
        print()
        print(f"Start the resolution with OWA weight = {owaWeights}")
        print(f"---------------------------------------------------")
    
    startTime = time()

    model, (_, fs, rs, bs) = initOWAModel(values, weights, capacity, owaWeights)
    zs = []

    model.optimize()

    while model.status == gp.GRB.OPTIMAL:
        y = tuple(round(fs[k].X) for k in range(n))
        lorenzVec = tuple(round((k + 1) * rs[k].X - sum(bs[k, i].X for i in range(n))) for k in range(n))

        if verbose:
            print(f"Iteration {it}:")
            print(f"    y              = {y}")
            print(f"    lorenz vector  = {lorenzVec}")
        if findAllLorenzND:
            similarNDPoints = findSimilar(y, lorenzVec, values, weights, capacity, owaWeights)
            lorenzNDPoints |= similarNDPoints
            if verbose:
                print(f"    similar points = {similarNDPoints}")
        else:
            lorenzNDPoints.add(y)
        it += 1

        zs.append(model.addVars(n, vtype=gp.GRB.BINARY))
        model.update()

        model.addConstrs((((k + 1) * rs[k] - gp.quicksum(bs[k, i] for i in range(n)) >= (lorenzVec[k] + 1) * zs[-1][k]) for k in range(n)))
        model.addConstr(gp.quicksum(zs[-1][k] for k in range(n)) >= 1)

        model.optimize()
    
    endTime = time()
    
    return {
        "pareto": set(lorenzNDPoints),
        "lorenz": lorenzNDPoints,
        "number-of-objectives": n,
        "number-of-items": m,
        "runtime": endTime - startTime
    }

if __name__ == "__main__":
    from dataIO import loadKPData

    df, capacity = loadKPData("../data/2KP200-TA-0.dat")

    m, n = 20, 2 # 20 objects and 2 objectives.

    df = df.head(m)
    colstrs = list(df.columns)

    weights = df["w"].to_numpy(dtype=int)
    values = df[colstrs[1 : n + 1]].to_numpy(dtype=int)
    capacity = int(np.floor(weights.sum() / 2))

    owaWeights = [2, 1]

    results = milpApproach(values, weights, capacity, owaWeights)
    print(results["lorenz"])
