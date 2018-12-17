__author__ = 'Ruslan N. Kosarev'

import os
from thirdparty import pywavefront as pw
import numpy as np

inpdir = os.path.join(os.path.pardir, 'data')

# ======================================================================================================================
if __name__ == '__main__':

    # read json file
    filename = os.path.join(inpdir, 'subject_01/Model/frontal1/obj/110920150452_2.obj')
    data = pw.Wavefront(filename, collect_faces=True)

    points = data.points
    colors = data.colors
    print('points', points.shape)
    print('colors', colors.shape)

    # normals
    normals = data.normals
    print('normals', normals.shape)

    # cells
    faces = data.faces
    print('faces', faces.shape)
