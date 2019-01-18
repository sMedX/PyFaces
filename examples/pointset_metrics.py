__author__ = 'Ruslan N. Kosarev'

import os
from thirdparty import pywavefront as pw
from tffaces.metrics import MovingToFixedPointSetMetrics, MeshToMeshMetrics
from tffaces.models import FaceModel
from examples import models, dirs
import open3d


def obj2open3dmesh(obj):
    mesh = open3d.TriangleMesh()
    mesh.vertices = open3d.Vector3dVector(obj.points)
    mesh.triangles = open3d.Vector3iVector(obj.faces)
    mesh.vertex_colors = open3d.Vector3dVector(obj.colors)
    if mesh.compute_vertex_normals() is False:
        mesh.compute_vertex_normals()
    return mesh


# ======================================================================================================================
if __name__ == '__main__':

    # read obj file
    filename = os.path.join(dirs.inpdir, 'subject_01/Model/frontal1/obj/110920150452_new.obj')
    obj = pw.Wavefront(filename)
    print('number of points', obj.number_of_points)
    print('number of faces', obj.number_of_faces)

    # read model face
    filename = models.bfm2017face12nomouth
    model = FaceModel(filename=filename)
    points = model.shape.points
    faces = model.shape.cells
    colors = model.color.colors
    print(model)

    # compute metrics
    metrics = MovingToFixedPointSetMetrics(moving=points, fixed=obj.points, registration=True)
    print(metrics)

    # compute mesh to mesh metrics
    fixed = obj2open3dmesh(obj)

    moving = open3d.TriangleMesh()
    moving.vertices = open3d.Vector3dVector(points)
    moving.triangles = open3d.Vector3iVector(faces)
    moving.vertex_colors = open3d.Vector3dVector(colors)
    moving.compute_vertex_normals()

    metrics = MeshToMeshMetrics(moving=moving, fixed=fixed, registration=1)
    print(metrics)
