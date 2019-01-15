__author__ = 'Ruslan N. Kosarev'

import os
from thirdparty import pywavefront as pw
from examples import dirs

image_height = 256
image_width = 256

# ======================================================================================================================
if __name__ == '__main__':

    # read obj file
    filename = os.path.join(dirs.inpdir, 'subject_01/Model/frontal1/obj/110920150452.obj')
    data = pw.Wavefront(filename)
    print('number of points', data.number_of_points)
    print('number of faces', data.number_of_faces)
