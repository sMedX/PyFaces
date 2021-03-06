__author__ = 'Ruslan N. Kosarev'

import os
from examples import models, dirs
from tffaces.detector import Dlib
from tffaces import landmarks
import numpy as np
import tensorflow as tf
from thirdparty.tf_mesh_renderer.mesh_renderer.rasterize_triangles import minimum_perspective_threshold as divide_threshold


# ======================================================================================================================
class AmbientColor:
    def __init__(self, value=(1, 1, 1), dtype=tf.float32):
        self.dtype = dtype
        self.value = np.expand_dims(np.array(value), axis=0)
        self.tensor = tf.Variable(self.value, dtype=self.dtype, name=self.name)

    @property
    def name(self):
        return self.__class__.__name__

    def update(self, input):
        if isinstance(input, tf.Session):
            input = input.run(self.tensor)

        self.value = input
        self.tensor = tf.Variable(self.value, dtype=self.dtype, name=self.name)


class CameraConfig:
    def __init__(self):
        # camera position
        position = np.array([[0, 0, 1000]], dtype=np.float32)
        self.position = tf.Variable(position, name='camera_position')

        look_at = np.array([[0, 0, 0]], dtype=np.float32)
        self.look_at = tf.Variable(look_at, name='camera_look_at')

        up = np.array([[0, 1, 0]], dtype=np.float32)
        self.up = tf.Variable(up, name='camera_up_direction')

        self.ambient_color = AmbientColor()

        self.fov_y = tf.constant([15.0], dtype=tf.float32)
        self.near_clip = tf.constant([0.1], dtype=tf.float32)
        self.far_clip = tf.constant([2000], dtype=tf.float32)

        self.divide_threshold = divide_threshold


class LightConfig:
    def __init__(self):
        positions = np.array([[[0, 0, 1000],
                               [0, 0, 1000],
                               [0, 0, 1000]]], dtype=np.float32)
        self.positions = tf.Variable(positions, name='light_positions')

        intensities = np.zeros([1, 3, 3], dtype=np.float32)
        self.intensities = tf.Variable(intensities, name='light_intensities')


# ======================================================================================================================
class BaselFaceModel2017NoMouthDlib:
    def __init__(self):
        # model and landmark detector
        self._model_file = models.bfm2017nomouth
        self.detector = Dlib()

        # landmark
        self.landmarks = landmarks.BaselFaceModeNoMouth2017Dlib()

        self.camera = CameraConfig()

        self.light = LightConfig()

    @property
    def model_file(self):
        return self._model_file


# ======================================================================================================================
class BaselFaceModel2017Face12Dlib:
    def __init__(self):
        # model and landmark detector
        self._model_file = models.bfm2017face12nomouth
        self.detector = Dlib()

        # landmark
        self.landmarks = landmarks.BaselFaceModeNoMouth2017Dlib()

        self.camera = CameraConfig()

        self.light = LightConfig()

    @property
    def model_file(self):
        return self._model_file


