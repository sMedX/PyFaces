__author__ = 'Ruslan N. Kosarev'

import os
from tffaces.models import FaceModel, ModelTransform
from tffaces.fit import ModelToImageLandmarkRegistration, ModelToImageColorRegistration
from tffaces import transforms
from tffaces import imutils
from examples import dirs, config

width = 300
iterations = 1000

# ======================================================================================================================
if __name__ == '__main__':

    config = config.BaselFaceModel2017Face12Dlib()

    # read image
    filename = os.path.join(dirs.inpdir, 'basel_face_example.png')
    image = imutils.read(filename, width=width)

    # read model face
    filename = config.model_file
    model = FaceModel(filename=filename, landmarks=config.landmarks)

    # initialize model transform
    transform = ModelTransform(model=model,
                               transform=transforms.SimilarityEuler3DTransform(),
                               bounds=(3, 3, None))

    fit = ModelToImageLandmarkRegistration(image=image,
                                           transform=transform,
                                           detector=config.detector,
                                           camera=config.camera)
    fit.run()
    fit.report()
    filename = os.path.join(dirs.outdir, 'fitlandmarks.png')
    fit.show(show=False, save=filename)

    # model to image registration
    fit = ModelToImageColorRegistration(image=image,
                                        transform=transform,
                                        camera=config.camera,
                                        light=config.light,
                                        iterations=iterations)
    fit.run()
    fit.report()
    filename = os.path.join(dirs.outdir, 'fitcolors.png')
    fit.show(show=True, save=filename)
