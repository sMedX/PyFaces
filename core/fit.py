__author__ = 'Ruslan N. Kosarev'

import matplotlib.pyplot as plt
import tensorflow as tf
from mesh_renderer import camera_utils


# data to represent surface
class ShapeToImageLandmarkRegistration:
    def __init__(self, model=None, image=None, detector=None, camera=None, transform=None):
        self.model = model
        self.image = image
        self.detector = detector

        self.transform = transform
        self.transform.center = self.model.shape.center

        self.camera = camera
        self.learning_rate = 0.01
        self.number_of_iterations = 5000

        self.outputs = None

    def detect_landmarks(self):
        return self.detector.detect_landmarks(self.image)

    def render_landmarks(self, points):

        height = self.image.shape[0]
        width = self.image.shape[1]

        # transform model points to image
        camera_matrices = camera_utils.look_at(self.camera.position, self.camera.look_at, self.camera.up)
        perspective_transform = camera_utils.perspective(width/height, self.camera.fov_y, self.camera.near_clip, self.camera.far_clip)
        transform = tf.matmul(perspective_transform, camera_matrices)

        points = points / self.camera.scale
        points = tf.concat((points, tf.ones([1, points.shape[1], 1], dtype=tf.float32)), axis=2)

        clip_space_points = tf.matmul(points, transform, transpose_b=True)
        clip_space_points_w = tf.maximum(tf.abs(clip_space_points[:, :, 3:4]), self.camera.divide_threshold) * \
                              tf.sign(clip_space_points[:, :, 3:4])

        normalized_device_coordinates = clip_space_points[:, :, 0:3] / clip_space_points_w

        x_image_coordinates = (normalized_device_coordinates[0, :, 0] + tf.constant(1, dtype=tf.float32)) * \
                              tf.constant(width / 2, dtype=tf.float32)
        x_image_coordinates = tf.expand_dims(x_image_coordinates, axis=1)

        y_image_coordinates = (tf.constant(1, dtype=tf.float32) - normalized_device_coordinates[0, :, 1]) * \
                              tf.constant(height / 2, dtype=tf.float32)
        y_image_coordinates = tf.expand_dims(y_image_coordinates, axis=1)

        image_coordinates = tf.concat((x_image_coordinates, y_image_coordinates), axis=1)

        return image_coordinates

    def run(self):
        # collect landmarks
        landmarks_indices = self.model.landmarks.get_weights() > 0.1
        image_landmarks = tf.constant(self.detect_landmarks()[landmarks_indices, :], dtype=tf.float32)
        model_landmarks = tf.constant(self.model.landmarks.to_array()[landmarks_indices, :], dtype=tf.float32)

        # apply spatial transform to model landmarks
        spatial_parameters = self.transform.variable_parameters
        transformed_landmarks = self.transform.transform(model_landmarks)
        transformed_landmarks = tf.expand_dims(transformed_landmarks, 0)

        # apply render
        rendered_landmarks = self.render_landmarks(transformed_landmarks)

        loss = tf.reduce_mean(tf.square(image_landmarks - rendered_landmarks))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        variables = spatial_parameters
        gradients, variables = zip(*optimizer.compute_gradients(loss, variables))
        train_step = optimizer.apply_gradients(list(zip(gradients, variables)))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for i in range(self.number_of_iterations):
            self.outputs = sess.run((train_step,                 # 0
                                     loss,                       # 1
                                     self.transform.parameters,  # 2
                                     rendered_landmarks          # 3
                                     ))

            if i == 0 or (i + 1) % 100 == 0 or (i+1) == self.number_of_iterations:
                print('-----------------------------------------------------------')
                print('iteration', i + 1, '(', self.number_of_iterations, '), loss', self.outputs[1])
                print('parameters', self.outputs[2])

    def show(self, show=True):
        landmarks = self.outputs[3]

        ax = self.detector.show()
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='.', s=5)

        if show is True:
            plt.show()

        return ax

