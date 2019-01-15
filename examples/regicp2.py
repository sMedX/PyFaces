__author__ = 'Ruslan N. Kosarev'

import os
import open3d
import numpy as np
import copy
from examples import dirs


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.draw_geometries([source_temp, target_temp])


def mesh2pcd(mesh):
    if mesh.compute_vertex_normals() is False:
        mesh.compute_vertex_normals()

    pcd = open3d.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals
    return pcd


# ======================================================================================================================
if __name__ == "__main__":
    tfile = os.path.join(dirs.inpdir, 'open3d/110920150452_new.ply')
    mesh = open3d.read_triangle_mesh(tfile)
    target = mesh2pcd(mesh)

    sfile = os.path.join(dirs.inpdir, 'open3d/mean_face.ply')
    mesh = open3d.read_triangle_mesh(sfile)
    source = mesh2pcd(mesh)

    open3d.draw_geometries([target])
    open3d.draw_geometries([source])

    trans_init = np.asarray([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = open3d.evaluate_registration(source, target, np.Inf, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = open3d.registration_icp(source, target, np.Inf, trans_init, open3d.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p.transformation)

    print("Apply point-to-plane ICP")
    reg_p2l = open3d.registration_icp(source, target, np.Inf, trans_init, open3d.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    print("")
    draw_registration_result(source, target, reg_p2l.transformation)