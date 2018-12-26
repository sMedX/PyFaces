__author__ = 'Ruslan N. Kosarev'

import os
from tffaces import transforms
from tffaces.models import FaceModel, ModelTransform
from tffaces import imutils
from tffaces.fit import ModelToImageLandmarkRegistration, ModelToImageColorRegistration, ModelToImageShapeRegistration
from examples import dirs, config

width = 300
number_of_epochs = 3
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
    print(model)

    # initialize model transform
    transform = ModelTransform(model=model,
                               transform=transforms.SimilarityEuler3DTransform(),
                               bounds=(1, 1, None))

    # model to image landmark based registration
    fit = ModelToImageLandmarkRegistration(image=image,
                                           transform=transform,
                                           detector=config.detector,
                                           camera=config.camera)
    fit.run()
    fit.report()
    fit.show(show=False, save='landmarks.png')

    # ------------------------------------------------------------------------------------------------------------------
    for epoch in range(1, number_of_epochs+1):
        print('---------------------------------------')
        print('epochs {}/{}'.format(epoch, number_of_epochs))

        transform.set_number_of_components((epoch/number_of_epochs, 0, None))

        # model to image registration
        fit = ModelToImageColorRegistration(image=image,
                                            transform=transform,
                                            camera=config.camera,
                                            light=config.light,
                                            iterations=iterations)
        fit.run()
        fit.report()
        filename = os.path.join(dirs.outdir, '{:02d}-color.png'.format(epoch))
        fit.show(show=False, save=filename)

        # model to image registration
        fit = ModelToImageShapeRegistration(image=image,
                                            transform=transform,
                                            camera=config.camera,
                                            light=config.light,
                                            iterations=iterations)
        fit.run()
        fit.report()
        filename = os.path.join(dirs.outdir, '{:02d}-shape.png'.format(epoch))
        fit.show(show=False, save=filename)
