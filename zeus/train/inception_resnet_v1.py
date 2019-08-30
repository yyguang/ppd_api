#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author: chenyunkuo 2019-03-11 10:53:14
# Ref: https://github.com/titu1994/Inception-v4/blob/master/inception_resnet_v1.py

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dropout, Dense, Lambda, Flatten, Activation, concatenate, add
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K

"""
Implementation of Inception-Residual Network v1 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.

Some additional details:
[1] Each of the A, B and C blocks have a 'scale_residual' parameter.
    The scale residual parameter is according to the paper. It is however turned OFF by default.

    Simply setting 'scale=True' in the create_inception_resnet_v1() method will add scaling.
"""

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = layers.BatchNormalization(axis=bn_axis,
                                      scale=False,
                                      name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)
    return x

def inception_resnet_stem(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    c = conv2d_bn(input, 32, 3, strides=2, padding='valid')
    c = conv2d_bn(c, 32, 3, padding='valid')
    c = conv2d_bn(c, 64, 3)
    c = MaxPooling2D(3, strides=2, padding='valid')(c)
    c = conv2d_bn(c, 80, 1)
    c = conv2d_bn(c, 192, 3, padding='valid')
    c = conv2d_bn(c, 256, 3, strides=2, padding='valid')
    return c

def inception_resnet_A(input, scale_residual=0.1):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = conv2d_bn(input, 32, 1)

    ir2 = conv2d_bn(input, 32, 1)
    ir2 = conv2d_bn(ir2, 32, 3)

    ir3 = conv2d_bn(input, 32, 1)
    ir3 = conv2d_bn(ir3, 32, 3)
    ir3 = conv2d_bn(ir3, 32, 3)

    ir_merge = concatenate([ir1, ir2, ir3], axis=channel_axis)

    ir_conv = conv2d_bn(ir_merge, 256, 1, activation='linear')
    ir_conv = Lambda(lambda x: x * scale_residual)(ir_conv)

    out = add([init, ir_conv])
    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)
    return out

def inception_resnet_B(input, scale_residual=0.1):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = conv2d_bn(input, 128, 1)

    ir2 = conv2d_bn(input, 128, 1)
    ir2 = conv2d_bn(ir2, 128, [1, 7])
    ir2 = conv2d_bn(ir2, 128, [7, 1])

    ir_merge = concatenate([ir1, ir2], axis=channel_axis)

    ir_conv = conv2d_bn(ir_merge, 896, 1, activation='linear')
    ir_conv = Lambda(lambda x: x * scale_residual)(ir_conv)

    out = add([init, ir_conv])
    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)
    return out

def inception_resnet_C(input, scale_residual=0.1):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = conv2d_bn(input, 192, 1)

    ir2 = conv2d_bn(input, 192, 1)
    ir2 = conv2d_bn(ir2, 192, [1, 3])
    ir2 = conv2d_bn(ir2, 192, [3, 1])

    ir_merge = concatenate([ir1, ir2], axis=channel_axis)

    ir_conv = conv2d_bn(ir_merge, 1792, 1, activation='linear')
    ir_conv = Lambda(lambda x: x * scale_residual)(ir_conv)

    out = add([init, ir_conv])
    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)
    return out

def reduction_A(input, k=192, l=224, m=256, n=384):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = MaxPooling2D(3, strides=2, padding='valid')(input)

    r2 = conv2d_bn(input, n, 3, strides=2, padding='valid')

    r3 = conv2d_bn(input, k, 1)
    r3 = conv2d_bn(r3, l, 3)
    r3 = conv2d_bn(r3, m, 3, strides=2, padding='valid')

    m = concatenate([r1, r2, r3], axis=channel_axis)
    m = BatchNormalization(axis=channel_axis)(m)
    m = Activation('relu')(m)
    return m


def reduction_resnet_B(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = MaxPooling2D(3, strides=2, padding='valid')(input)

    r2 = conv2d_bn(input, 256, 1)
    r2 = conv2d_bn(r2, 384, 3, strides=2, padding='valid')

    r3 = conv2d_bn(input, 256, 1)
    r3 = conv2d_bn(r3, 256, 3, strides=2, padding='valid')

    r4 = conv2d_bn(input, 256, 1)
    r4 = conv2d_bn(r4, 256, 3)
    r4 = conv2d_bn(r4, 256, 3, strides=2, padding='valid')

    m = concatenate([r1, r2, r3, r4], axis=channel_axis)
    m = BatchNormalization(axis=channel_axis)(m)
    m = Activation('relu')(m)
    return m

def inception_resnet_v1(nb_classes=1001, scale=(0.17, 0.1, 0.2), include_top=True):
    '''
    Creates a inception resnet v1 network

    :param nb_classes: number of classes.txt
    :param scale: flag to add scaling of activations
    :param include_top: whether to include the fully-connected
            layer at the top of the network.
    :return: Keras Model with 1 input (299x299x3) input shape and 2 outputs (final_output, auxiliary_output)
    '''
    if K.image_dim_ordering() == 'th':
        init = Input((3, 299, 299))
    else:
        init = Input((299, 299, 3))

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_resnet_stem(init)

    # 5 x Inception Resnet A
    for i in range(5):
        x = inception_resnet_A(x, scale_residual=scale[0])

    # Reduction A - From Inception v4
    x = reduction_A(x, k=192, l=192, m=256, n=384)

    # 10 x Inception Resnet B
    for i in range(10):
        x = inception_resnet_B(x, scale_residual=scale[1])
    
    '''
    # Auxiliary tower
    aux_out = AveragePooling2D((5, 5), strides=(3, 3))(x)
    aux_out = Convolution2D(128, 1, 1, border_mode='same', activation='relu')(aux_out)
    aux_out = Convolution2D(768, 5, 5, activation='relu')(aux_out)
    aux_out = Flatten()(aux_out)
    aux_out = Dense(nb_classes, activation='softmax')(aux_out)
    '''

    # Reduction Resnet B
    x = reduction_resnet_B(x)

    # 5 x Inception Resnet C
    for i in range(5):
        x = inception_resnet_C(x, scale_residual=scale[2])

    if include_top:
      # Average Pooling
      x = AveragePooling2D((8,8))(x)

      # Dropout
      x = Dropout(0.8)(x)
      x = Flatten()(x)

      # Output
      x = Dense(output_dim=nb_classes, activation='softmax')(x)

    model = Model(init, output=x, name='inception_resnet_v1')

    return model

if __name__ == "__main__":
    from tensorflow.keras.utils.visualize_util import plot_model

    inception_resnet_v1 = inception_resnet_v1()

    plot_model(inception_resnet_v1, to_file="Inception ResNet-v1.png", show_shapes=True)

