__author__ = 'Ruslan N. Kosarev'

import os
from core import transforms
from core.models import FaceModel, ModelTransform
from core import imutils
from core.fit import ModelToImageLandmarkRegistration, ModelToImageRegistration
import config

scale = 0.5

# ======================================================================================================================
if __name__ == '__main__':

    config = config.BaselFaceModeNoMouth2017Dlib()

    # read image
    filename = os.path.join(os.path.pardir, 'data', 'basel_face_example.png')
    image = imutils.read(filename, scale=scale, show=False)

    # read model face
    filename = config.model_file
    model = FaceModel(filename=filename, landmarks=config.landmarks)
    print(model)

    # initialize model transform
    transform = ModelTransform(model=model,
                               transform=transforms.SimilarityEuler3DTransform())

    # ------------------------------------------------------------------------------------------------------------------
    # model to image landmark based registration
    fit = ModelToImageLandmarkRegistration(image=image,
                                           model=model,
                                           detector=config.detector,
                                           camera=config.camera,
                                           transform=transform.spatial_transform)
    fit.run()
    fit.show()

    # ------------------------------------------------------------------------------------------------------------------
    # model to image registration
    fit = ModelToImageRegistration(image=image,
                                   transform=transform,
                                   camera=config.camera,
                                   light=config.light)
    fit.run()
    fit.show()
