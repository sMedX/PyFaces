__author__ = 'Ruslan N. Kosarev'

import os
from core import transforms
from core.models import FaceModel, ModelTransform
from core import imutils
from core.fit import ModelToImageLandmarkRegistration
import config

inpdir = os.path.join(os.path.pardir, 'data')
outdir = os.path.join(os.path.pardir, 'output')

width = 500
iterations = 1000

# ======================================================================================================================
if __name__ == '__main__':

    config = config.BaselFaceModel2017Face12Dlib()

    # read image
    # filename = os.path.join(inpdir, 'basel_face_example.png')
    # filename = os.path.join(inpdir, 'example_01.jpg')
    filename = os.path.join(inpdir, 'example_02.jpg')
    image = imutils.read(filename, width=width)

    # read model face
    filename = config.model_file
    model = FaceModel(filename=filename, landmarks=config.landmarks)
    print(model)

    # initialize model transform
    transform = ModelTransform(model=model,
                               transform=transforms.SimilarityEuler3DTransform(),
                               bounds=(None, None, None))

    # model to image landmark based registration
    fit = ModelToImageLandmarkRegistration(image=image,
                                           transform=transform,
                                           detector=config.detector,
                                           camera=config.camera)
    fit.run()
    fit.report()
    fit.show(show=True, save='landmarks.png')
