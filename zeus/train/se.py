#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <teng.cao@foxmail.com> 2018-09-17 23:11:23
#https://github.com/titu1994/keras-squeeze-excite-network/blob/master/se.py
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K


def squeeze_excite_block(input, weight_decay=0.0, ratio=16, name=None):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    if name == None:
        name = "se_block"
    if weight_decay != 0.0:
        kernel_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False, kernel_regularizer=kernel_regularizer, name=name + "_dense0")(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False, kernel_regularizer=kernel_regularizer, name=name + "_dense1")(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x
