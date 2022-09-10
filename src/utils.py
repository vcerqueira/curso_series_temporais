import itertools
from typing import Dict


def expand_grid(*iters):
    product = list(itertools.product(*iters))
    return {i: [x[i] for x in product]
            for i in range(len(iters))}


def expand_grid_all(x: Dict) -> Dict:
    param_grid = expand_grid(*x.values())
    new_keys = dict(zip(param_grid.keys(), x.keys()))

    param_grid = {new_keys[k]: v for k, v in param_grid.items()}

    return param_grid
