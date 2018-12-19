__author__ = 'Ruslan N. Kosarev'

import os
from plyfile import PlyData
import numpy as np

inpdir = os.path.join(os.path.pardir, 'data')

# ======================================================================================================================
if __name__ == '__main__':

    # read json file
    filename = os.path.join(inpdir, 'subject_01/Model/frontal1/obj/110920150452_2.ply')
    with open(filename, 'rb') as f:
        ply = PlyData.read(f)

    print(ply.elements[0].name)
    print(ply.elements[0][0])
    x=1
