import pandas as pd
import re

def readKPData(filename: str) -> tuple[pd.DataFrame, int]:
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

if __name__ == "__main__":
    print(readKPData("../data/2KP200-TA-0.dat"))
    