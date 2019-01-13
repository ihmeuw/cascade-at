from itertools import product
from math import nan

import numpy as np
import pandas as pd


class Var:
    """A Var is a set of values of a random field on a SmoothGrid."""
    def __init__(self, age_time_grid, count=1):
        self.ages = np.array(age_time_grid[0], dtype=np.float)
        self.times = np.array(age_time_grid[1], dtype=np.float)
        age_time = np.array(list(product(sorted(self.ages), sorted(self.times))))
        self.grid = pd.DataFrame(dict(
            age=np.tile(age_time[:, 0], count),
            time=np.tile(age_time[:, 1], count),
            mean=nan,
            idx=np.repeat(range(count), len(age_time)),
        ))
        self.mulstd = dict()  # keys are value, dage, dtime.

    @property
    def age_time(self):
        return (self.ages, self.times)

    def __len__(self):
        return self.ages.shape[0] * self.times.shape[0] + len(self.mulstd)

    def __str__(self):
        return f"Var({len(self.ages), len(self.times)})"
