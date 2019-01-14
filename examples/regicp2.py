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


# ======================================================================================================================
if __name__ == "__main__":
    tfile = os.path.join(dirs.inpdir, 'open3d/110920150452_new.ply')
    target = open3d.read_point_cloud(tfile)

    sfile = os.path.join(dirs.inpdir, 'open3d/mean_face.ply')
    source = open3d.read_point_cloud(sfile)

    open3d.draw_geometries([target])
    open3d.draw_geometries([source])

    threshold = 1000

    trans_init = np.asarray(
                [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = open3d.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = open3d.registration_icp(source, target, threshold, trans_init,
                                      open3d.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p.transformation)
