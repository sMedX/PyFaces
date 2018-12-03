__author__ = 'Ruslan N. Kosarev'

import os
from core.models import FaceModel, ModelTransform
from core.fit import ModelToImageLandmarkRegistration, ModelToImageShapeRegistration
from core import transforms
from core import imutils
import config

inpdir = os.path.join(os.path.pardir, 'data')
outdir = os.path.join(os.path.pardir, 'output')

width = 300
iterations = 1000

# ======================================================================================================================
if __name__ == '__main__':

    config = config.BaselFaceModel2017Face12Dlib()

    # read image
    filename = os.path.join(inpdir, 'basel_face_example.png')
    image = imutils.read(filename, width=width)

    # read model face
    filename = config.model_file
    model = FaceModel(filename=filename, landmarks=config.landmarks)

    # initialize model transform
    transform = ModelTransform(model=model,
                               transform=transforms.SimilarityEuler3DTransform(),
                               bounds=(2, 2, None))

    fit = ModelToImageLandmarkRegistration(image=image,
                                           transform=transform,
                                           detector=config.detector,
                                           camera=config.camera)
    fit.run()
    fit.report()
    filename = os.path.join(outdir, 'fitlandmarks.png')
    fit.show(show=False, save=filename)

    # model to image registration
    fit = ModelToImageShapeRegistration(image=image,
                                        transform=transform,
                                        camera=config.camera,
                                        light=config.light,
                                        iterations=iterations)
    fit.run()
    fit.report()
    filename = os.path.join(outdir, 'fitcolors.png')
    fit.show(show=True, save=filename)
