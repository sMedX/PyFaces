__author__ = 'Ruslan N. Kosarev'

import tensorflow as tf


def tfSession():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    return session
