#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author: chenyunkuo 2019-05-30 15:38:35

import os
import sys
import imghdr
import random
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from sm_sequence import sm_load_img
from tensorflow.keras import backend as K
from keras_preprocessing.image.utils import img_to_array
from sm_sequence import sm_load_img, read_random_crop_dict


# 读取图片路径和标签
def read_datalist(filename, image_size, preprocessing_function):
  all_image_paths, all_image_labels = [], []
  with open(filename, 'r') as f:
    for line in f:
      parts = line.strip().split()
      assert len(parts) == 2
      #img, crop_result = sm_load_img(path=parts[0], target_size=(image_size, image_size))
      #x = img_to_array(img, data_format=K.image_data_format())
      #x = preprocessing_function(x)
      if imghdr.what(parts[0]) not in {"jpeg", "png"}:
        continue
      all_image_paths.append(parts[0])
      all_image_labels.append(parts[1])
  classes = sorted(set(all_image_labels))
  class_num = len(classes)
  class_indices = dict(zip(classes, range(len(classes))))
  all_image_onehotlabels = []
  classes = []
  for label in all_image_labels:
    onehot = [0] * class_num
    label_idx = class_indices[label]
    classes.append(label_idx)
    onehot[label_idx] = 1
    all_image_onehotlabels.append(onehot)
  return all_image_paths, all_image_onehotlabels, class_indices, classes


# 图像前处理
def parse_image_puretf(img_path, label, image_size, preprocessing_function):
  image = tf.read_file(img_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [image_size, image_size])
  #img, crop_result = sm_load_img(img_path, target_size=(image_size, image_size))
  #x = img_to_array(img, data_format=K.image_data_format())
  x = preprocessing_function(image)
  return x, label



# 创建tf.Dataset
def SMDataset(directory, batch_size, image_size, preprocessing_function, num_parallel_calls=8, buffer_size=100000, is_train=True, flags=None, random_crop=0, aspect_ratio_range=(0.75, 1.33),
                     area_range=(0.05, 1.0), *args, **kwargs):

  # pure tensorflow.dataset
  pure_tf = False
  
  random_crop_dict = {}
  if random_crop:
    random_crop_dict = read_random_crop_dict(flags.random_crop_list)

  def parse_image(img_path, label, image_size=image_size, preprocessing_function=preprocessing_function, is_train=is_train, flags=flags, random_crop=random_crop, aspect_ratio_range=aspect_ratio_range,
                     area_range=area_range, random_crop_dict=random_crop_dict):
    #image = tf.read_file(img_path)
    #image = tf.image.decode_jpeg(image, channels=3)
    #image = tf.image.resize_images(image, [image_size, image_size])
    if is_train:
      img, crop_result = sm_load_img(img_path,
                                     data_format=K.image_data_format(),
                                     target_size=(image_size, image_size),
                                     interpolation=flags.interpolation,
                                     rotation_range=flags.rotation_range,
                                     rotation90=flags.rotation90,
                                     horizontal_flip=flags.horizontal_flip,
                                     vertical_flip=flags.vertical_flip,
                                     letter_box=flags.letter_box,
                                     levels=flags.zoom_pyramid_levels,
                                     short_side=flags.zoom_short_side,
                                     hflip=random.randint(0, int(flags.horizontal_flip)),
                                     random_crop=random_crop,
                                     bbox_data=random_crop_dict.get(os.path.basename(img_path), []),
                                     min_object_covered=flags.min_object_covered,
                                     aspect_ratio_range=aspect_ratio_range,
                                     area_range=area_range,
                                     gaussian_blur=flags.gaussian_blur,
                                     motion_blur=flags.motion_blur,
                                     brightness=flags.brightness,
                                     saturation=flags.saturation)
    else:
      img, crop_result = sm_load_img(img_path,
                                     data_format=K.image_data_format(),
                                     target_size=(image_size, image_size),
                                     interpolation='bilinear',
                                     letter_box=flags.letter_box)

    x = img_to_array(img, data_format=K.image_data_format())
    x = preprocessing_function(x)
    return x, label

  all_image_paths, all_image_labels, class_indices, classes = read_datalist(directory, image_size, preprocessing_function)
  
  def set_tf_shape(x, y):
    x.set_shape((image_size, image_size, 3))
    y.set_shape((len(class_indices), ))
    return x, y
  
  dataset = Dataset.from_tensor_slices((all_image_paths, all_image_labels))
  if pure_tf:
    img_label_ds = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True).apply( \
      tf.contrib.data.map_and_batch(map_func=(lambda x, y: parse_image_puretf(x, y, image_size, preprocessing_function)),
                                             batch_size=batch_size, num_parallel_calls=num_parallel_calls))


  else:
    img_label_ds = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True).map(
                   lambda x, y: tf.py_func(parse_image, [x, y], [tf.float32, y.dtype])).map(
                   lambda x, y: set_tf_shape(x, y)).batch(batch_size)

  img_label_ds = img_label_ds.prefetch(buffer_size=batch_size)
  #img_label_ds = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True).batch(batch_size)
  img_label_iterator = img_label_ds.make_initializable_iterator()
  img_label_iterator.len = len(all_image_paths) / float(batch_size)
  img_label_iterator.filenames = all_image_paths
  img_label_iterator.classes = classes
  img_label_iterator.class_indices = class_indices
  img_label_iterator.batch_size = batch_size
  img_label_iterator.image_size = image_size
  return img_label_iterator
