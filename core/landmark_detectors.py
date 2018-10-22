__author__ = 'Ruslan N. Kosarev'

import dlib
import cv2
import numpy as np
from thirdparty.facial_landmarks.imutils import face_utils
from thirdparty.facial_landmarks import imutils


def dlib_detector(image, shape_file):

    # load the input image, resize it, and convert it to grayscale
    # image = imutils.resize(image, width=500)

    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_file)

    # detect faces in the gray scale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    points = np.zeros([0, 2])

    # loop over the face detections
    for (i, rect) in enumerate(rects):

        # determine the facial landmarks for the face region, then convert the facial landmarks to a NumPy array
        shape = face_utils.shape_to_np(predictor(gray, rect))
        shape[:, 1] = image.shape[0] - shape[:,1]

        # concatenate coordinates
        points = np.concatenate((points, shape), axis=0)

    return points