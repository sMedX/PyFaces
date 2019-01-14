__author__ = 'Ruslan N. Kosarev'

import os
import open3d
import numpy as np
import copy
from thirdparty import pywavefront as pw
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
    source = open3d.read_point_cloud(os.path.join(dirs.inpdir, 'open3d/cloud_bin_0.pcd'))
    target = open3d.read_point_cloud(os.path.join(dirs.inpdir, 'open3d/cloud_bin_1.pcd'))
    threshold = 0.02
    trans_init = np.asarray(
                [[0.862, 0.011, -0.507,  0.5],
                [-0.139, 0.967, -0.215,  0.7],
                [0.487, 0.255,  0.835, -1.4],
                [0.0, 0.0, 0.0, 1.0]])

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

    print("Apply point-to-plane ICP")
    reg_p2l = open3d.registration_icp(source, target, threshold, trans_init,
                                      open3d.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    print("")
    draw_registration_result(source, target, reg_p2l.transformation)