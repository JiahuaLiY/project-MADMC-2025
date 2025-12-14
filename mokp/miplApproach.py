import numpy as np
import gurobipy as gp

# from time import time

def initOWAModel(values: np.ndarray, weights: np.ndarray, capacity: int, owaWeights: np.ndarray) -> tuple[gp.Model, tuple[gp.tupledict[int, gp.Var]]]:
    """"""
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
    """Find all non dominated points with same lorenz vector as y."""
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
        y = tuple(fs[k].X for k in range(n))
        similarNDPoints.add(y)

        zs.append(model.addVars(n, vtype=gp.GRB.BINARY))
        model.update()

        model.addConstrs(((fs[i] >= (y[i] + 1) * zs[-1][i]) for i in range(n)))
        model.addConstr(gp.quicksum(zs[-1][k] for k in range(n)) >= 1)

        model.optimize()
    
    return similarNDPoints

def miplApproach(values: np.ndarray, weights: np.ndarray, capacity: int, owaWeights: np.ndarray, findAllLorenzND: bool=False) -> dict[str, set[tuple]]:
    """"""
    lorenzNDPoints = set()
    _, n = values.shape

    model, (_, fs, rs, bs) = initOWAModel(values, weights, capacity, owaWeights)
    zs = []

    model.optimize()

    while model.status == gp.GRB.OPTIMAL:
        y = tuple(fs[k].X for k in range(n))
        lorenzVec = tuple((k + 1) * rs[k].X - sum(bs[k, i].X for i in range(n)) for k in range(n))

        if findAllLorenzND:
            lorenzNDPoints |= findSimilar(y, lorenzVec, values, weights, capacity, owaWeights)
        else:
            lorenzNDPoints.add(y)

        zs.append(model.addVars(n, vtype=gp.GRB.BINARY))
        model.update()

        model.addConstrs((((k + 1) * rs[k] - gp.quicksum(bs[k, i] for i in range(n)) >= (lorenzVec[k] + 1) * zs[-1][k]) for k in range(n)))
        model.addConstr(gp.quicksum(zs[-1][k] for k in range(n)) >= 1)

        model.optimize()
    
    return { "pareto": set(), "lorenz": lorenzNDPoints }

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

    results = miplApproach(values, weights, capacity, owaWeights)
    print(results["lorenz"])
