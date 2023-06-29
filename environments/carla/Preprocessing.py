"""
Master Thesis project by Artur Niederfahrenhorst
This file defines our data preprocessing.
"""

import tensorflow as tf


def preprocess_images(image, configuration):
    """
    Depending on the configuration, crop, grayscale and shift image mean.
    :param image: RBG image
    :param configuration: configuration object
    :return: preprocessed image
    """
    color_dim = 3
    if configuration['CROPIMAGE']:
        image = image[:, configuration['FROM_Y']:configuration['TO_Y'], configuration['FROM_X']:configuration['TO_X'], :]
    # we crop side of screen as they carry little information
    else:
        image = image
    if configuration['MEANSHIFT']:
        image = tf.subtract(image, 127)
        image = tf.divide(image, 127.0)
    else:
        image = tf.divide(image, 255.0)
    if configuration['CROPIMAGE']:
        image = tf.reshape(image, [-1, configuration['TO_Y'] - configuration['FROM_Y'],
                                   configuration['TO_X'] - configuration['FROM_X'], color_dim])
        return image
    else:
        image = tf.reshape(image, [-1,
                                   configuration['STATE_DIM'][0],
                                   configuration['STATE_DIM'][1],
                                   configuration['STATE_DIM'][2]])
        return image
