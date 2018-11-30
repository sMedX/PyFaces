__author__ = 'Ruslan N. Kosarev'

import os
from core.detector import Dlib
from core import landmarks
import numpy as np
import tensorflow as tf
from mesh_renderer.rasterize_triangles import MINIMUM_PERSPECTIVE_DIVIDE_THRESHOLD as divide_threshold


# ======================================================================================================================
class Data:
    def __init__(self):
        self.inpdir = os.path.abspath('data')
        self.outdir = os.path.abspath('output')


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
        position = np.array([[0, 0, 5]], dtype=np.float32)
        self.position = tf.Variable(position, name='camera_position')

        look_at = np.array([[0, 0, 0]], dtype=np.float32)
        self.look_at = tf.Variable(look_at, name='camera_look_at')

        up = np.array([[0, 1, 0]], dtype=np.float32)
        self.up = tf.Variable(up, name='camera_up_direction')

        self.ambient_color = AmbientColor()

        self.fov_y = tf.constant([30.0], dtype=tf.float32)
        self.near_clip = tf.constant([0.01], dtype=tf.float32)
        self.far_clip = tf.constant([10.0], dtype=tf.float32)

        self.divide_threshold = divide_threshold
        self.scale = 100


class LightConfig:
    def __init__(self):
        positions = np.array([[[0, 0, 50], [0, 0, 50], [0, 0, 50]]], dtype=np.float32)
        self.positions = tf.Variable(positions, name='light_positions')

        intensities = np.zeros([1, 3, 3], dtype=np.float32)
        self.intensities = tf.Variable(intensities, name='light_intensities')


# ======================================================================================================================
class BaselFaceModeNoMouth2017Dlib:
    def __init__(self):
        # model and landmark detector
        self._model_file = 'model2017-1_bfm_nomouth.h5'
        self.detector = Dlib()

        # landmark
        self.landmarks = landmarks.BaselFaceModeNoMouth2017Dlib()

        self.camera = CameraConfig()

        self.light = LightConfig()

    @property
    def model_file(self):
        return os.path.join(os.path.pardir, 'data', self._model_file)

