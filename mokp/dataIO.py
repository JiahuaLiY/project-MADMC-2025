import pandas as pd
import re
import csv

def loadKPData(filename: str) -> tuple[pd.DataFrame, int]:
    """Read the knapsack problem data."""
    pattern = "c[ \t]+w[ \t]+(v[0-9]+[ \t]+)+(v[0-9]+)*"
    colformat = None

    capacity = None
    data = []

    with open(filename, "r") as file:
        for line in file:
            if colformat is None and re.match(pattern, line):
                dataStrs = line.split()
                colformat = dataStrs[1:]
            
            if line[0] == "i":
                dataStrs = line.split()
                data.append([
                    int(dataStrs[i]) for i in range(1, len(dataStrs))
                ])
            elif line[0] == "W":
                dataStrs = line.split()
                capacity = int(dataStrs[1])
    
    assert colformat is not None
    assert capacity is not None and capacity > 0
    return pd.DataFrame(data, columns=colformat), capacity

def saveData(filename: str, data: dict) -> None:
    """"""
    m = data["number-of-items"]
    n = data["number-of-objectives"]
    paretoNDPoints = data["pareto"]
    lorenzNDPoints = data["lorenz"]
    runtime = data["runtime"]

    with open(filename, "w") as file:
        file.write(f"[nb-items] {m}\n")
        file.write(f"[nb-objectives] {n}\n")
        file.write(f"[nb-pareto] {len(paretoNDPoints)}\n")
        file.write(f"[nb-lorenz] {len(lorenzNDPoints)}\n")
        file.write(f"[runtime] {runtime}\n")

        file.write("[format] ")
        file.write(" ".join(f"f{i + 1}" for i in range(n)))
        file.write(" isLorenzND\n")

        for y in lorenzNDPoints:
            file.write("[vector] ")
            file.write(" ".join(map(str, y)))
            file.write(" True\n")
        
        for y in paretoNDPoints - lorenzNDPoints:
            file.write("[vector] ")
            file.write(" ".join(map(str, y)))
            file.write(" False\n")

def loadData(filename) -> dict:
    """"""
    data = { "pareto": set(), "lorenz": set() }
    with open(filename, "r") as file:
        for line in file:
            dataStrs = line.split()
            print(dataStrs)
            if dataStrs[0] == "[nb-items]":
                data["number-of-items"] = int(dataStrs[1])
            elif dataStrs[0] == "[nb-objectives]":
                data["number-of-objectives"] = int(dataStrs[1])
            elif dataStrs[0] == "[runtime]":
                data["runtime"] = float(dataStrs[1])
            elif dataStrs[0] == "[vector]":
                isLorenzND = bool(dataStrs[-1])
                if isLorenzND:
                    data["lorenz"].add(tuple(float(dataStrs[i]) for i in range(1, len(dataStrs) - 2)))
                data["pareto"].add(tuple(float(dataStrs[i]) for i in range(1, len(dataStrs) - 2)))
    return data

if __name__ == "__main__":
    print(loadKPData("../data/2KP200-TA-0.dat"))
    