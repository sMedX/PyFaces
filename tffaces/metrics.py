__author__ = 'Ruslan N. Kosarev'

from scipy.spatial import cKDTree as KDTree
import numpy as np
import time
# from numba import jit


# point set metrics
class MovingToFixedPointSetMetrics:
    def __init__(self, moving, fixed, registration=False):
        self.moving_points = moving
        self.fixed_points = fixed
        self.rmsd = None
        self.mean = None
        self.elapsed_time = None

        self.registration = registration
        self.transform = None

        # perform registration
        if self.registration is True:
            self._perform_registration()

        # compute metrics
        self._compute()

    def __repr__(self):
        """Representation of metric"""
        return (
            '\n{}\n'.format(self.__class__.__name__) +
            'mean {:.6f}\n'.format(self.mean) +
            'rmsd {:.6f}\n'.format(self.rmsd) +
            'elapsed time {:.3f}\n'.format(self.elapsed_time)
        )

    @property
    def number_of_moving_points(self):
        return self.moving_points.shape[0]

    @property
    def number_of_fixed_points(self):
        return self.fixed_points.shape[0]

    def _perform_registration(self):
        import open3d

        def array2pcd(xyz):
            pcd = open3d.PointCloud()
            pcd.points = open3d.Vector3dVector(xyz)
            return pcd

        fixed = array2pcd(self.fixed_points)
        moving = array2pcd(self.moving_points)

        init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        regp2p = open3d.registration_icp(moving, fixed, np.Inf, init, open3d.TransformationEstimationPointToPoint())
        self.transform = regp2p.transformation

        print(regp2p)

        moving_points = np.concatenate((self.moving_points, np.ones((self.number_of_moving_points, 1))), axis=1)
        self.moving_points = moving_points @ np.transpose(self.transform)[:, :3]

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


# point set metrics
import open3d


def open3dmeshopen3dpcd(mesh):
    if mesh.compute_vertex_normals() is False:
        mesh.compute_vertex_normals()

    pcd = open3d.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals
    return pcd


class MeshToMeshMetrics:
    def __init__(self, moving, fixed, registration=2):
        self.moving = moving
        self.fixed = fixed
        self.rmsd = None
        self.mean = None
        self.elapsed_time = None

        self.registration = registration
        self.transform = None

        # perform registration
        self._perform_registration()

        # compute metrics
        self._compute()

    def __repr__(self):
        """Representation of metric"""
        return (
            '\n{} (registration: {})\n'.format(self.__class__.__name__, self.registration_name) +
            'mean {:.6f}\n'.format(self.mean) +
            'rmsd {:.6f}\n'.format(self.rmsd) +
            'elapsed time {:.3f}\n'.format(self.elapsed_time)
        )

    @property
    def number_of_moving_points(self):
        return self.moving_points.shape[0]

    @property
    def number_of_fixed_points(self):
        return self.fixed_points.shape[0]

    @property
    def fixed_points(self):
        if type(self.fixed) is type(open3d.TriangleMesh()):
            return np.asarray(self.fixed.vertices)

    @property
    def moving_points(self):
        if type(self.fixed) is type(open3d.TriangleMesh()):
            return np.asarray(self.moving.vertices)

    def _perform_registration(self):

        if self.registration == 0:
            self.registration_name = 'None'
            return

        fixed = open3dmeshopen3dpcd(self.fixed)
        moving = open3dmeshopen3dpcd(self.moving)

        if self.registration == 1:
            estimator = open3d.TransformationEstimationPointToPoint()
        else:
            estimator = open3d.TransformationEstimationPointToPlane()

        init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        registration = open3d.registration_icp(moving, fixed, np.Inf, init, estimator)

        self.registration_name = type(estimator)
        self.transform = registration.transformation
        self.moving.transform(self.transform)

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
