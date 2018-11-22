__author__ = 'Ruslan N. Kosarev'

import numpy as np
from math import sin, cos, pi
import tensorflow as tf


# data to represent surface
class ShapeToImageLandmarkRegistration:
    def __init__(self, model=None, image=None, detector=None):
        self._model = model
        self._image = image
        self._detector = detector

        self.number_of_iterations = 100

    @property
    def image(self):
        return self._image

    @property
    def detector(self):
        return self._detector

    def detect_landmarks(self, show=False):
        points = self.detector.detect_landmarks(self.image)

        if show is True:
            self.detector.show()

        return points

    def run(self):
        image_landmarks = tf.Variable(self.detect_landmarks(), dtype=tf.float32)
        model_landmarks = tf.Variable(self.detect_landmarks(), dtype=tf.float32)

        loss = tf.reduce_mean(tf.square(image_landmarks - model_landmarks))

        pass