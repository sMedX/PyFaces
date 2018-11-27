__author__ = 'Ruslan N. Kosarev'

import os
from core.models import FaceModel
from core.fit import ModelToImageLandmarkRegistration
from core import transforms
from core import imutils
import config

scale = 0.5

# ======================================================================================================================
if __name__ == '__main__':

    config = config.BaselFaceModeNoMouth2017Dlib()

    # read image
    filename = os.path.join(os.path.pardir, 'data', 'basel_face_example.png')
    image = imutils.read(filename, scale=scale)

    # read model face
    filename = config.model_file
    model = FaceModel(filename=filename, landmarks=config.landmarks)

    # spatial transform
    transform = transforms.SimilarityEuler3DTransform()

    fit = ModelToImageLandmarkRegistration(image=image,
                                           model=model,
                                           detector=config.detector,
                                           camera=config.camera,
                                           transform=transform)
    fit.run()
    fit.show()

