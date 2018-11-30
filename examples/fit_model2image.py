__author__ = 'Ruslan N. Kosarev'

import os
from core import transforms
from core.models import FaceModel, ModelTransform
from core import imutils
from core.fit import ModelToImageLandmarkRegistration, ModelToImageColorRegistration, ModelToImageShapeRegistration
import config

inpdir = os.path.join(os.path.pardir, 'data')
outdir = os.path.join(os.path.pardir, 'output')

scale = 0.5
number_of_epochs = 50

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
                               transform=transforms.SimilarityEuler3DTransform(),
                               bounds=(3, 3, None))

    # ------------------------------------------------------------------------------------------------------------------
    # model to image landmark based registration
    fit = ModelToImageLandmarkRegistration(image=image,
                                           transform=transform,
                                           detector=config.detector,
                                           camera=config.camera)
    fit.run()
    filename = os.path.join(outdir, 'landmarks.png')
    fit.show(show=False, save=filename)

    for epoch in range(number_of_epochs):
        print('---------------------------------------')
        print('epochs {}/{}'.format(epoch, number_of_epochs))

        # model to image registration
        fit = ModelToImageColorRegistration(image=image,
                                            transform=transform,
                                            camera=config.camera,
                                            light=config.light,
                                            iterations=200)
        fit.run()
        filename = os.path.join(outdir, '{:02d}-color.png'.format(epoch))
        fit.show(show=False, save=filename)

        # model to image registration
        fit = ModelToImageShapeRegistration(image=image,
                                            transform=transform,
                                            camera=config.camera,
                                            light=config.light,
                                            iterations=500)
        fit.run()
        filename = os.path.join(outdir, '{:02d}-shape.png'.format(epoch))
        fit.show(show=False, save=filename)
