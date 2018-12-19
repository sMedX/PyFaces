__author__ = 'Ruslan N. Kosarev'

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from thirdparty import pywavefront as pw
from thirdparty import mesh_renderer
import config

image_height = 256
image_width = 256

inpdir = os.path.join(os.path.pardir, 'data')
outdir = os.path.join(os.path.pardir, 'output')

# ======================================================================================================================
if __name__ == '__main__':

    # read obj file
    filename = os.path.join(inpdir, 'subject_01/Model/frontal1/obj/110920150452_new.obj')
    data = pw.Wavefront(filename)
    print('number of points', data.number_of_points)
    print('number of faces', data.number_of_faces)

    # initialize renderer
    points = tf.constant(np.expand_dims(data.points, axis=0), dtype=tf.float32)
    cells = tf.constant(data.faces, dtype=tf.int32)
    normals = tf.constant(np.expand_dims(data.normals, axis=0), dtype=tf.float32)
    colors = tf.constant(np.expand_dims(data.colors, axis=0), dtype=tf.float32)

    render = mesh_renderer.initialize(config.CameraConfig(),
                                      config.LightConfig(),
                                      points,
                                      cells,
                                      normals,
                                      colors,
                                      image_width,
                                      image_height)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    output = sess.run([render])

    # show rendered image
    image = output[0][0, :, :, :3]
    mask = output[0][0, :, :, 3]

    print('minimal value', np.min(image))
    print('maximal value', np.max(image))

    # save image to file
    filename = os.path.splitext(os.path.basename(filename))[0] + '.png'
    filename = os.path.join(outdir, filename)
    cv2.imwrite(filename, cv2.cvtColor(255*image, cv2.COLOR_RGB2BGR))

    # show outputs
    plt.imshow(image)
    plt.show()