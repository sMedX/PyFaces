__author__ = 'Ruslan N. Kosarev'

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from thirdparty.tf_mesh_renderer.mesh_renderer import mesh_renderer
import cv2
import json
from examples import dirs

height = 512
width = 512

# ======================================================================================================================
if __name__ == '__main__':

    # read json file
    filename = os.path.join(dirs.inpdir, 'subject_01/Model/frontal1/obj/110920150452_new.json')
    with open(filename) as f:
        data = json.load(f)

    print(data.keys())

    # points
    print(data['vertices'][0]['name'])
    points = data['vertices'][0]['values']
    number_of_points = int(len(points) / 3)
    points = np.reshape(points, [number_of_points, 3])
    print('points', points.shape)

    # normals
    print(data['vertices'][1]['name'])
    normals = data['vertices'][1]['values']
    normals = np.reshape(normals, [number_of_points, 3])
    print('normals', normals.shape)

    # colors
    print(data['vertices'][2]['name'])
    colors = data['vertices'][2]['values']
    colors = np.reshape(colors, [int(len(colors) / 4), 4])[:, :3]
    print('colors', colors.shape)

    # cells
    print(data['connectivity'][0]['name'])
    cells = data['connectivity'][0]['indices']
    number_of_cells = int(len(cells) / 3)
    cells = np.reshape(cells, [number_of_cells, 3])
    print('cells', cells.shape)

    # camera position
    camera_position = np.array([[0, 0, 1000]], dtype=np.float32)
    camera_position = tf.Variable(camera_position, name='camera_position')

    camera_look_at = np.array([[0, 0, 0]], dtype=np.float32)
    camera_look_at = tf.Variable(camera_look_at, name='camera_look_at')

    camera_up = np.array([[0, 1, 0]], dtype=np.float32)
    camera_up = tf.Variable(camera_up, name='camera_up_direction')

    # light positions and light intensities
    light_positions = np.array([[[0, 0, 1000],
                                 [0, 0, 1000],
                                 [0, 0, 1000]]], dtype=np.float32)
    light_positions = tf.Variable(light_positions, name='light_positions')

    fov_y = tf.constant([15.0], dtype=tf.float32)
    near_clip = tf.constant([0.01], dtype=tf.float32)
    far_clip = tf.constant([2000.0], dtype=tf.float32)

    light_intensities = np.zeros([1, 3, 3], dtype=np.float32)
    light_intensities = tf.Variable(light_intensities, name='light_intensities')
    ambient_color = tf.Variable(np.ones([1, 3]), dtype=tf.float32)

    # initialize renderer
    points = tf.constant(np.expand_dims(points, axis=0), dtype=tf.float32)
    cells = tf.constant(cells, dtype=tf.int32)
    normals = tf.constant(np.expand_dims(normals, axis=0), dtype=tf.float32)
    colors = tf.constant(np.expand_dims(colors/255, axis=0), dtype=tf.float32)

    renderer = mesh_renderer(
        points,
        cells,
        normals,
        colors,
        camera_position,
        camera_look_at,
        camera_up,
        light_positions,
        light_intensities,
        width,
        height,
        ambient_color=ambient_color,
        fov_y=fov_y,
        near_clip=near_clip,
        far_clip=far_clip
    )

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    output = sess.run([renderer])

    # show rendered image
    image = output[0][0, :, :, :3]
    mask = output[0][0, :, :, 3]

    print('minimal value', np.min(image))
    print('maximal value', np.max(image))

    # save image to file
    filename = os.path.splitext(os.path.basename(filename))[0] + '.png'
    filename = os.path.join(dirs.outdir, filename)
    cv2.imwrite(filename, cv2.cvtColor(255*image, cv2.COLOR_RGB2BGR))

    # show outputs
    plt.imshow(image)
    plt.show()


