__author__ = 'Ruslan N. Kosarev'

from scipy.spatial.kdtree import KDTree
import numpy as np
import time
# from numba import jit


# point set metrics
class MovingToFixedPointSetMetrics:
    def __init__(self, moving, fixed):
        self.moving_points = moving
        self.fixed_points = fixed
        self.rmsd = None
        self.mean = None
        self.elapsed_time = None

        # compute metrics
        self._compute()

    def __repr__(self):
        """Representation of metric"""
        return (
            '{}\n'.format(self.__class__.__name__) +
            'mean {:.3f}\n'.format(self.mean) +
            ' rms {:.3f}\n'.format(self.rmsd) +
            'elapsed time {:.3f}\n'.format(self.elapsed_time)
        )

    @property
    def number_of_moving_points(self):
        return self.moving_points.shape[0]

    @property
    def number_of_fixed_points(self):
        return self.fixed_points.shape[0]

    def _compute(self):
        start_time = time.time()

        tree = KDTree(self.fixed_points)

        self.mean = 0
        self.rmsd = 0

        for point in self.moving_points:
            dist, index = tree.query(point)
            self.mean += dist
            self.rmsd += pow(dist, 2)

        self.mean = self.mean/self.number_of_moving_points
        self.rmsd = np.sqrt(self.rmsd/self.number_of_moving_points)

        self.elapsed_time = time.time() - start_time
