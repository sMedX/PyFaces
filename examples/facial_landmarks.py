__author__ = 'Ruslan N. Kosarev'

# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

import argparse
import os
import matplotlib.pyplot as plt
import dlib
import cv2
import numpy as np
from thirdparty.facial_landmarks.imutils import face_utils
from thirdparty.facial_landmarks import imutils


# ======================================================================================================================
if __name__ == '__main__':

    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
    # ap.add_argument("-i", "--image", required=True,	help="path to input image")
    # args = vars(ap.parse_args())

    image_file = 'example_01.jpg'
    image_file = os.path.join(os.path.pardir, 'data', image_file)

    # shape model file
    shape_file = 'shape_predictor_68_face_landmarks.dat'
    shape_file = os.path.join(os.path.pardir, 'data', shape_file)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_file)
    image = imutils.resize(image, width=500)

    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_file)

    # detect faces in the gray scale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    xx = []
    yy = []

    # loop over the face detections
    for (i, rect) in enumerate(rects):

        # determine the facial landmarks for the face region, then convert the facial landmarks to a NumPy array
        shape = face_utils.shape_to_np(predictor(gray, rect))

        # concatenate coordinates
        xx = np.concatenate((xx, shape[:, 0]))
        yy = np.concatenate((yy, shape[:, 1]))

    # show the output image with the face detections + facial landmarks
    fig, ax = plt.subplots()
    im = ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.scatter(xx, yy, c='r', marker='.', s=5)
    plt.show()