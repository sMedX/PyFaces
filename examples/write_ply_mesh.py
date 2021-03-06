__author__ = 'Ruslan N. Kosarev'

import os
import numpy as np
from plyfile import PlyData, PlyElement
from tffaces.models import FaceModel
import itertools
from examples import models, joinoutdir

# ======================================================================================================================
if __name__ == '__main__':

    # read model face
    model = FaceModel(filename=models.bfm2017nomouth)
    points = model.shape.points
    colors = model.color.colors
    print(model)

    # initialize vertexes
    vertexes = [tuple(itertools.chain(*t, [255])) for t in zip(points, 255*colors)]
    vertexes = np.array(vertexes, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')])
    vertexes = PlyElement.describe(vertexes, 'vertex')

    # initialize faces
    faces = [(f,) for f in np.transpose(model.shape.representer.np_cells)]
    faces = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])
    faces = PlyElement.describe(faces, 'face')

    # write output ply file
    filename = os.path.splitext(os.path.basename(model.filename))[0] + '.ply'
    filename = joinoutdir(filename)
    PlyData((vertexes, faces), text=True).write(filename)
