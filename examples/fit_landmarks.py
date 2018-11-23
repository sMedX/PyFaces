__author__ = 'Ruslan N. Kosarev'

import os
import cv2
from core.models import FaceModel
from core.fit import ShapeToImageLandmarkRegistration
from core import transforms
import config

scale = 0.5

# ======================================================================================================================
if __name__ == '__main__':

    config = config.BaselFaceModeNoMouth2017Dlib()

    # read image
    image_file = 'basel_face_example.png'
    image_file = os.path.join(os.path.pardir, 'data', image_file)

    image = cv2.imread(image_file)
    height = int(scale*image.shape[0])
    width = int(scale*image.shape[1])
    image = cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # read model face
    filename = os.path.join(os.path.pardir, 'data', config.model_file)
    model = FaceModel(filename=filename, landmarks=config.landmarks)

    # spatial transform
    transform = transforms.SimilarityEuler3DTransform()

    fit = ShapeToImageLandmarkRegistration(image=image,
                                           model=model,
                                           detector=config.detector,
                                           camera=config.camera,
                                           transform=transform)
    fit.run()
    fit.show()

