__author__ = 'Ruslan N. Kosarev'

import os
from thirdparty import pywavefront as pw
from tffaces.metrics import MovingToFixedPointSetMetrics
from tffaces.models import FaceModel
from examples import models, dirs

height = 512
width = 512

# ======================================================================================================================
if __name__ == '__main__':

    # read obj file
    filename = os.path.join(dirs.inpdir, 'subject_01/Model/frontal1/obj/110920150452_new.obj')
    data = pw.Wavefront(filename)
    print('number of points', data.number_of_points)
    print('number of faces', data.number_of_faces)

    # read model face
    filename = models.bfm2017face12nomouth
    model = FaceModel(filename=filename)
    points = model.shape.points
    colors = model.color.colors
    print(model)

    # compute metrics
    metrics = MovingToFixedPointSetMetrics(moving=points, fixed=data.points, registration=2)
    print(metrics)
