__author__ = 'Ruslan N. Kosarev'

import matplotlib.pyplot as plt
import os
import dlib
import numpy as np
import cv2
from thirdparty.facial_landmarks.imutils import face_utils


class Dlib:
    def __init__(self):
        shape_file = 'shape_predictor_68_face_landmarks.dat'
        self._shape_file = os.path.join(os.path.pardir, 'data', shape_file)

        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(self._shape_file)

        self._points = None
        self._image = None

    @property
    def points(self):
        return self._points

    @property
    def image(self):
        return self._image

    def detect_landmarks(self, image):
        self._image = image

        # detect faces in the gray scale image
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        rects = self._detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then convert the facial landmarks to a NumPy array
            self._points = face_utils.shape_to_np(self._predictor(gray, rect))
            break

        return self._points

    def show(self, show=True):
        # show the output image with the face detections
        fig, ax = plt.subplots()
        ax.imshow(self.image)
        ax.scatter(self.points[:, 0], self.points[:, 1], c='g', marker='*', s=5)

        for count, point in enumerate(self.points):
            plt.text(point[0] + 1, point[1] + 1, '{}'.format(count), color='blue', fontsize=7)

        if show is True:
            plt.show()

        return fig, ax
