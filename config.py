__author__ = 'Ruslan N. Kosarev'

import os
from core.detector import Dlib
from core import landmarks
import numpy as np
import tensorflow as tf
from mesh_renderer.rasterize_triangles import MINIMUM_PERSPECTIVE_DIVIDE_THRESHOLD as divide_threshold


class BaselFaceModeNoMouth2017Dlib:
    def __init__(self):
        # model and landmark detector
        self._model_file = 'model2017-1_bfm_nomouth.h5'
        self.detector = Dlib()

        # landmark
        self.landmarks = landmarks.BaselFaceModeNoMouth2017Dlib()

        class CameraConfig:
            def __init__(self):
                # camera position
                position = np.array([0, 0, 5], dtype=np.float32)
                position = tf.Variable(position, name='camera_position')
                tf.expand_dims(position, axis=0)
                self.position = tf.tile(tf.expand_dims(position, axis=0), [1, 1])

                look_at = np.array([0, 0, 0], dtype=np.float32)
                look_at = tf.Variable(look_at, name='camera_look_at')
                self.look_at = tf.tile(tf.expand_dims(look_at, axis=0), [1, 1])

                up = np.array([0, 1, 0], dtype=np.float32)
                up = tf.Variable(up, name='camera_up_direction')
                self.up = tf.tile(tf.expand_dims(up, axis=0), [1, 1])

                self.fov_y = tf.constant([30.0], dtype=tf.float32)
                self.near_clip = tf.constant([0.01], dtype=tf.float32)
                self.far_clip = tf.constant([10.0], dtype=tf.float32)

                self.divide_threshold = divide_threshold
                self.scale = 100

        self.camera = CameraConfig()

    @property
    def model_file(self):
        return os.path.join(os.path.pardir, 'data', self._model_file)

