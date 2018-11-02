"""
Defines demography functions.
"""
import numpy as np

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


class DemographicInterval:
    """
    An ordered set of age intervals, consecutive from some starting age.
    We often need the width of each interval, or the start of intervals,
    their finish times, or the boundaries of all intervals.
    This also compares intervals to create new ones.

    Attributes:
        nx (np.ndarray): Widths of intervals

        bound (np.ndarray): Endpoints of intervals. The length is
            one longer than the length of nx.

        start (np.ndarray): Array of starting ages.

        finish (np.ndarray): Array of finishing ages for each interval.

        omega (np.double): Oldest possible age in these intervals.
    """

    def __init__(self, nx, begin=0):
        """

        Args:
            nx (List[float]|np.ndarray): Width of each interval
            begin (float): Starting age, defaults to 0.
        """
        nx = np.array(nx, dtype=np.double)
        CODELOG.debug(f"di nx {nx} begin {begin} {nx.cumsum()}")
        self.bound = np.hstack([[begin], begin + nx.cumsum()])
        self.nx = nx

    @property
    def start(self):
        return self.bound[:-1]

    @property
    def finish(self):
        return self.bound[1:]

    @property
    def omega(self):
        return self.finish[-1]

    def __getitem__(self, key):
        """Returns a new DemographicInterval subset. Still continuous."""
        if isinstance(key, slice):
            a = key.start or 0
            b = key.stop or len(self.nx)
        else:
            a, b = key, key + 1
        CODELOG.debug(f"di getitem keystart {a} {b}")
        return DemographicInterval(self.nx[a:b], begin=self.bound[a])

    def overlaps_with(self, other):
        """All age groups in this interval that overlap the other intervals."""
        eps = 1e-6  # Accounts for two intervals nearly lining up.
        past_left = np.where(self.finish > other.start[0] + eps)[0]
        before_right = np.where(self.start < other.finish[-1] - eps)[0]
        return self.__getitem__(slice(past_left[0], before_right[-1] + 1))

    def __len__(self):
        return self.nx.shape[0]

    def __str__(self):
        if len(self.nx) > 2:
            return f"({self.start[0]} {self.start[1]} {self.start[2]} {self.finish[-1]})"
        else:
            return f"DemographicInterval({self.nx.shape})"

    def __repr__(self):
        return f"DemographicInterval({self.nx})"
