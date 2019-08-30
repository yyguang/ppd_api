#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <cao-teng-teng.cao@foxmail.com> 2017-10-11 12:13:51

from __future__ import print_function
import os
import sys
import pwd
import math
import argparse
import shutil
import notebook_util
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, add, Lambda, Conv2D, LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import load_model
from tensorflow.python.keras.applications import keras_modules_injection
from sm_sequence import SMSequence
from sm_dataset import SMDataset
from roc_callback import RocCallback

# input arguments
FLAGS = None

# default model, name same as model.name from keras.
# Ref https://github.com/keras-team/keras-applications for model performance
INCEPTION_V3="inception_v3"
RESNET50="resnet50"
RESNET50_V2="resnet50v2"
RESNEXT50="resnext50"
XCEPTION="xception"
MOBILENET="mobilenet_1.00_224"
MOBILENETV2="mobilenetv2_1.40_224"
INCEPTION_RESNET_V1="inception_resnet_v1"
INCEPTION_RESNET_V2="inception_resnet_v2"
NASNET_LARGE="NASNet"
DENSENET201="densenet201"

# save the labels define txt file
def save_labels_file(filename, class_indices):
  labels = [''] * len(class_indices)
  for key, value in class_indices.items():
    assert value >= 0 & value < len(class_indices)
    labels[value] = key
  output_dir = os.path.dirname(filename)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  file_labels = open(filename, 'w')
  for label in labels:
    file_labels.write(label + '\n')
  file_labels.close()

# get the data generator for train or test
def get_generator(preprocess_input, path_data, list_mode,
    image_size, interpolation, letter_box, batch_size, random_crop, random_seed, flags, is_train, val_sequence=None):
  aspect_ratio_range = tuple(map(float, flags.aspect_ratio_range.split(',')))
  area_range = tuple(map(float, flags.area_range.split(',')))
 
  if flags.io_pipeline == 'dataset':
    return SMDataset(directory=path_data, 
                     batch_size=batch_size,
                     image_size=image_size,
                     preprocessing_function=preprocess_input,
                     is_train=is_train,
                     flags=flags,
                     random_crop=random_crop,
                     aspect_ratio_range=aspect_ratio_range,
                     area_range=area_range)
 
  if is_train:
    generator = SMSequence(
      directory=path_data,
      list_mode=list_mode,
      batch_size=batch_size,
      image_size=image_size,
      interpolation=interpolation,
      letter_box=letter_box,
      val_sequence=val_sequence,
      preprocessing_function=preprocess_input,
      optimizing_type=flags.optimizing_type,
      class_indices=flags.class_indices,
      label_smoothing=flags.label_smoothing,
      class_aware_sampling=flags.class_aware_sampling,
      shuffle=True,
      random_seed=random_seed,
      rotation90=flags.rotation90,
      rotation_range=flags.rotation_range,
      horizontal_flip=flags.horizontal_flip,
      vertical_flip=flags.vertical_flip,
      zoom_pyramid_levels=flags.zoom_pyramid_levels,
      zoom_short_side=flags.zoom_short_side,
      brightness=flags.brightness,
      saturation=flags.saturation,
      random_crop=random_crop,
      min_object_covered=flags.min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      random_crop_list=flags.random_crop_list,
      gaussian_blur=flags.gaussian_blur,
      motion_blur=flags.motion_blur,
      save_to_dir=flags.save_to_dir,
      save_prefix=flags.save_prefix,
      save_samples_num=flags.save_samples_num,
      mixup=flags.mixup,
      mixup_alpha=flags.mixup_alpha)
  else:
    generator = SMSequence(
      directory=path_data,
      list_mode=list_mode,
      batch_size=batch_size,
      image_size=image_size,
      interpolation='bilinear', # 预测的resize interpolation选择bilinear
      letter_box=letter_box,
      preprocessing_function=preprocess_input,
      optimizing_type=flags.optimizing_type,
      class_indices=flags.class_indices,
      shuffle=False)
  return generator

def get_generator_steps(train_generator, validation_generator, test_generator, flags):
  steps_train = train_generator.__len__() if flags.io_pipeline == 'sequence' else train_generator.len
  steps_validation = validation_generator.__len__() if flags.io_pipeline == 'sequence' else validation_generator.len
  steps_test = test_generator.__len__() if flags.io_pipeline == 'sequence' else test_generator.len
  return (steps_train, steps_validation, steps_test)

# calculate lrs for each stage
def init_lr_list(flags, epochs):
  lr_str = flags.learning_rate.split(';')
  lr_list_tmp = []
  for item in lr_str:
    if item.startswith('['):
      item = item[1:-1]
      lr_list_tmp.append(map(float, item.split(',')))
    else:
      lr_list_tmp.append([float(item)])

  lr_list = []
  for stage_idx, lrs in enumerate(lr_list_tmp):
    epoch = epochs[stage_idx]
    # 学习率为列表
    if len(lrs) > 1:
      if len(lrs) >= epoch:
        lrs_new = lrs[0:epoch]
      else:
        lrs_new = lrs[:]
        lrs_new.extend([lrs[-1]] * (epoch - len(lrs_new)))
    # 学习率为单值
    else:
      if flags.lr_decay:
        lrs_new = []
        for idx in range(epoch):
          lrs_new.append(lrs[0] * (flags.lr_decay_decayrate ** (idx / flags.lr_decay_decaystep)))
      else:
        lrs_new = [lrs[0]] * epoch
    assert len(lrs_new) == epoch
    lr_list.append(lrs_new)

  return lr_list

# lr schedule
def lr_schedule_callback(lrs):
  def step_decay(epoch):
    assert epoch < len(lrs)
    lr = lrs[epoch]
    # print epoch idx start from 1
    print('lr for epoch %d is %f' % (epoch+1, lr))
    return lr
  from tensorflow.keras.callbacks import LearningRateScheduler
  return LearningRateScheduler(step_decay, verbose=1)

# get preprocess_input for different models
def get_preprocess_input(model):
  if model.name == INCEPTION_V3:
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    print('preprocess of inception v3 is used')
    return preprocess_input
  elif model.name == RESNET50:
    from tensorflow.keras.applications.resnet50 import preprocess_input
    print('preprocess of resnet50 is used')
    return preprocess_input
  elif model.name == RESNET50_V2:
    from keras_applications.resnet_v2 import preprocess_input
    @keras_modules_injection
    def preprocess_input_internal(*args, **kwargs):
      return preprocess_input(*args, **kwargs)
    print('preprocess of resnet_v2 is used')
    return preprocess_input_internal
  elif model.name == RESNEXT50:
    from keras_applications.resnext import preprocess_input
    @keras_modules_injection
    def preprocess_input_internal(*args, **kwargs):
      return preprocess_input(*args, **kwargs)
    print('preprocess of resnext50 is used')
    return preprocess_input_internal
  elif model.name == XCEPTION:
    from tensorflow.keras.applications.xception import preprocess_input
    print('preprocess of xception is used')
    return preprocess_input
  elif model.name == MOBILENET or model.name == MOBILENETV2:
    from tensorflow.keras.applications.mobilenet import preprocess_input
    print('preprocess of mobilenet is used')
    return preprocess_input
  elif model.name == INCEPTION_RESNET_V1:
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    print('preprocess of inception resnet v2 is used')
    return preprocess_input
  elif model.name == INCEPTION_RESNET_V2:
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    print('preprocess of inception resnet v2 is used')
    return preprocess_input
  elif model.name == NASNET_LARGE:
    from tensorflow.keras.applications.nasnet import preprocess_input
    print('preprocess of nasnet is used')
    return preprocess_input
  elif model.name == DENSENET201:
    from tensorflow.keras.applications.densenet import preprocess_input
    print('preprocess of densenet is used')
    return preprocess_input
  else:
    print("There is no preprocess_input supported for " + model.name)
    sys.exit(-1)
    return None
 
# get image size for different models
def get_image_size(model):
  image_size = 299
  if model.name == INCEPTION_V3 or model.name == XCEPTION or model.name == INCEPTION_RESNET_V2 or model.name == INCEPTION_RESNET_V1:
    pass 
  elif model.name == RESNET50 or model.name == DENSENET201 or model.name == RESNET50_V2:
    image_size = 224
  elif model.name == MOBILENET or model.name == MOBILENETV2 or model.name == RESNEXT50:
    image_size = 224
  elif model.name == NASNET_LARGE:
    image_size = 331
  else:
    print("%s not support for auto inference image size" % model.name)
    sys.exit(-1)
  return image_size 

# config trainable layers for selected network
def set_trainable(finetune_blocks, model, quickly_ft):
  # first: train only the top layers (which were randomly initialized)
  if model.name == INCEPTION_V3:
    # i.e. freeze all convolutional InceptionV3 layers
    # total 10 blocks of inception blocks: 3 + 5 + 2
    block_to_layeridx = {0: 310, 1: 279, 2: 248, 3: 228, 
      4: 196, 5: 164, 6: 132, 7: 100,
      8: 86, 9: 63, 10: 40, 11: -1}
  elif model.name == RESNET50:
    # todo: more finetune layers from paper
    block_to_layeridx = {0: 173, 1: -1}
  elif model.name == RESNET50_V2:
    block_to_layeridx = {0: 189, 1: 153, 2: 85, 3: 39, 4: -1}
  elif model.name == RESNEXT50:
    block_to_layeridx = {0: 238, 1: 194, 2: 108, 3: 50, 4: -1}
  elif model.name == XCEPTION:
    # total 14 blocks of blocks:
    # 1. Entry flow: with 4 blocks
    # 2. Middle flow: with 8 repeated blocks
    # 3. Exit flow: with 2 blocks
    block_to_layeridx = {0: 131, 1: 125,
      2: 115, 3: 105, 4: 95, 5: 85, 6: 75, 7: 65, 8: 55, 9: 45,
      10: 35, 11: 25, 12: 15, 13: 6, 14: -1}
  elif model.name == MOBILENET:
    block_to_layeridx = {0: 95, 1: 88,
                         2: 81, 3: 74, 4: 67, 5: 60, 6: 53, 7: 46, 8: 39, 9: 32,
                         10: 25, 11: 18, 12: 11, 13: 4, 14: -1}
  elif model.name == MOBILENETV2:
    block_to_layeridx = {0:149, 1:146, 
                         2:138, 3:129, 4:120, 5:112, 6:103, 7:94, 8:86, 9:77,
                         10:68, 11:59, 12:51, 13:42, 14:33, 15:25, 16:17, 17:8, 18:-1}
  elif model.name == INCEPTION_RESNET_V1:
    block_to_layeridx = {0:182, 1:72, 2:11, 3:-1}
  elif model.name == INCEPTION_RESNET_V2:
    block_to_layeridx = {0: 779, 1: 776,
                         2: 761, 3: 745, 4: 729, 5: 713, 6: 697, 7: 681, 8: 665, 9: 649,
                         10: 633, 11: 594, 12: 578, 13: 562, 14: 546, 15: 530, 16: 514, 17: 498, 18: 482,     
                         19: 466, 20: 450, 21: 434, 22: 418, 23: 402, 24: 386, 25: 370, 26: 354, 27: 338,     
                         28: 322, 29: 306, 30: 290, 31: 260, 32: 238, 33: 216, 34: 194, 35: 172, 36: 150,     
                         37: 128, 38: 106, 39:  84, 40:  62, 41:  -1}     
  elif model.name == DENSENET201: 
    block_to_layeridx = {0: 705, 1: 704, 2: 476, 3: 136, 4: 48, 5: 4, 6: -1}    
  elif model.name == NASNET_LARGE:     
    block_to_layeridx = {0: 991, 1: 971,     
                         2: 947, 3: 902, 4: 857, 5: 812, 6: 767, 7: 722, 8: 676, 9: 631,     
                         10: 586, 11: 541, 12: 496, 13: 451, 14: 360, 15: 315, 16: 270, 17: 225, 18: 180,     
                         19: 135, 20: 92, 21: 64, 22: 47, 23: 18, 24: 3, 25: -1}
  else:
    print('finetune does not support: ' + model.name)
    sys.exit(-1)
  
  if finetune_blocks != -1 and (not block_to_layeridx.has_key(finetune_blocks)):
    print("%s not support finetune %d layers" % (model.name, finetune_blocks))
    sys.exit(-1)

  if finetune_blocks == -1:
    idx = -1
  else:
    idx = block_to_layeridx[finetune_blocks]
  layer_not_trainable = 0
  layer_trainable = 0
  # 快速调优模型，只设置softmax layer为trainable
  if quickly_ft == 1:
    idx = 312
  for layer in model.layers[:idx+1]:
    layer.trainable = False
    layer_not_trainable += 1
  for layer in model.layers[idx+1:]:
    layer.trainable = True
    layer_trainable += 1
  print('layer_not_trainable: %d, layer_trainable: %d' % (layer_not_trainable, layer_trainable))

# config optimizer for training
def set_optimizer(optimizer, lr, clipnorm):
  from tensorflow.keras.optimizers import SGD, Adam, RMSprop
  if optimizer == 'sgd':
    opt = SGD(lr=lr, momentum=0.9, clipnorm=clipnorm)
  elif optimizer == 'adam':
    opt = Adam(lr=lr, clipnorm=clipnorm)
  elif optimizer == 'rmsprop':
    opt = RMSprop(lr=lr, clipnorm=clipnorm)
  else:
    print('specified optimizer is not supported: ' + optimizer)
    print('supported optimizers are: sgd/adam/rmsprop')
    sys.exit(-1)
  return opt

# config callbacks
def set_callbacks(train_validation_files, flags, train_generator,
    test_generator, validation_generator, lrs, stage, history_metrics, workers):
  callbacks = [RocCallback(train_generator=train_generator,
        test_generator=test_generator,
        validation_generator=validation_generator,
        train_validation_files=train_validation_files,
        classes_map=flags.classes_map,
        multiprocessing=flags.multiprocessing,
        workers=workers,
        validation_auc=flags.validation_auc,
        test_auc=flags.test_auc,
        history_metrics=history_metrics, early_stopping_by_val_acc=flags.early_stopping_by_val_acc), lr_schedule_callback(lrs)]
  from multi_modelcheckpoint_callback import MultiModelCheckpoint
  monitor = 'val_acc' if flags.early_stopping_by_val_acc else 'val_loss'
  ckp_str = os.path.join(flags.output_dir,
      'model.stage%d.{epoch:02d}-{val_loss:.4f}.hdf5' % (stage+1))
  save_callback = MultiModelCheckpoint(ckp_str, monitor=monitor,
    save_best_only=True, save_weights_only=False, period=1, verbose=1)
  callbacks.append(save_callback)

  if flags.early_stopping != 0:
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=flags.early_stopping, verbose=1)
    callbacks.append(early_stopping)

  return callbacks

# config model
def set_model(model_name, num_classes, weight_decay, image_size, se_block, rgb_tsf_alpha, classify=True):
  # create the base pre-trained model
  assert(image_size == 0)
  if model_name == INCEPTION_V3:
    if se_block == 2:
      print('SE block is used for %s' % model_name)
    if weight_decay != 0:
      print('weight decay is used for %s' % model_name)
    if rgb_tsf_alpha:
      assert rgb_tsf_alpha in {0.1, 0.5}
      print('RGB transformations with leaky relu alpha %f is used for %s' % (rgb_tsf_alpha, model_name))
    #from keras.applications.inception_v3 import InceptionV3
    from inception_v3 import InceptionV3
    @keras_modules_injection
    def InceptionV3Internal(*args, **kwargs):
      return InceptionV3(*args, **kwargs)
    base_model = InceptionV3Internal(weights='imagenet', include_top=False,
        weight_decay=weight_decay, se_block=se_block, rgb_tsf_alpha=rgb_tsf_alpha)
    #base_model = InceptionV3(weights='imagenet', include_top=False)
  elif model_name == RESNET50:
    from tensorflow.keras.applications.resnet50 import ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False)
  elif model_name == RESNET50_V2:
    from keras_applications.resnet_v2 import ResNet50V2
    @keras_modules_injection
    def ResNet50V2Internal(*args, **kwargs):
      return ResNet50V2(*args, **kwargs)
    base_model = ResNet50V2Internal(weights='imagenet', include_top=False)
  elif model_name == RESNEXT50:
    from keras_applications.resnext import ResNeXt50
    @keras_modules_injection
    def ResNeXt50Internal(*args, **kwargs):
      return ResNeXt50(*args, **kwargs)
    base_model = ResNeXt50Internal(weights='imagenet', include_top=False, input_shape=(224,224,3))
  elif model_name == XCEPTION:
    from tensorflow.keras.applications.xception import Xception
    base_model = Xception(weights='imagenet', include_top=False)
  elif model_name == MOBILENET:
    from tensorflow.keras.applications.mobilenet import MobileNet
    base_model = MobileNet(alpha=1.0, weights='imagenet', include_top=False)
  elif model_name == MOBILENETV2:
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    base_model = MobileNetV2(alpha=1.4, weights='imagenet', include_top=False)
  elif model_name == INCEPTION_RESNET_V1:
    from inception_resnet_v1 import inception_resnet_v1
    base_model = inception_resnet_v1(include_top=False)
  elif model_name == INCEPTION_RESNET_V2:
    from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)
  elif model_name == DENSENET201:
    from tensorflow.keras.applications.densenet import DenseNet201
    base_model = DenseNet201(weights='imagenet', include_top=False)
  elif model_name == NASNET_LARGE:
    from tensorflow.keras.applications.nasnet import NASNetLarge
    base_model = NASNetLarge(weights=None, include_top=False)
    base_model.load_weights("/home/{}/.keras/models/nasnet_large_no_top.h5".format(pwd.getpwuid(os.getuid())[0]), by_name=True)
  elif model_name.endswith(".hdf5"):
    if os.path.exists(model_name):
      model = load_model(model_name)
      image_size = get_image_size(model)
      preprocess_input = get_preprocess_input(model)
      return (model, image_size, preprocess_input)
    else:
      print("Model " + model_name + " does not exist")
      sys.exit(-1)
  else:
    print("Model for " + model_name +  " is not supported")
    sys.exit(-1)
  print('base model of %s is used' % base_model.name)

  # use recommend image size for different models
  if image_size == 0:
    image_size = get_image_size(base_model)
    print('image size of %s is used' % image_size)
  preprocess_input = get_preprocess_input(base_model)
  
  # add a global spatial average pooling layer
  x = base_model.output
  if se_block == 1:
    from se import squeeze_excite_block
    x = squeeze_excite_block(x, weight_decay)
  if FLAGS.pooling == "avg":
    x = GlobalAveragePooling2D()(x)
  elif FLAGS.pooling == "max":
    x = GlobalMaxPooling2D()(x)
  elif FLAGS.pooling == "max-avg":
    a = GlobalMaxPooling2D()(x)
    b = GlobalAveragePooling2D()(x)
    x = add([a, b])
    x = Lambda(lambda z: 0.5 * z)(x)
  else:
    print("Mistake value of pooling with {}.".format(FLAGS.pooling))
    sys.exit(-1)
  if FLAGS.dense_units != 0:
    # let's add a fully-connected layer
    x = Dense(FLAGS.dense_units, activation='relu',
        kernel_regularizer=regularizers.l2(weight_decay))(x)
    #x = Dense(FLAGS.dense_units, activation='relu')(x)
  # and a logistic layer -- let's say we have 200 classes
  if classify:
    predictions = Dense(num_classes, activation='softmax',
      kernel_regularizer=regularizers.l2(weight_decay))(x)
  else:
    predictions = Dense(1, activation='linear', 
      kernel_regularizer=regularizers.l2(weight_decay))(x)

  # this is the model we will train
  from tensorflow.keras.models import Model
  model = Model(inputs=base_model.input, outputs=predictions, name=base_model.name)
  print('base_model layers: %d' % len(base_model.layers))

  #model.summary()
  #sys.exit(0)
  return (model, image_size, preprocess_input)

# load best model from last stage
def load_best_model(model_name, model_dir, stage, best_model_idx):
  # choose best model
  best_model = ''
  for filename in os.listdir(model_dir):
    if filename.endswith('.hdf5') and filename.startswith('model.stage%d.%02d' % (stage+1, best_model_idx+1)):
      best_model = filename
      break
  assert best_model != ''
  
  # load best model
  print('load best model from stage %d: %s' % (stage+1, best_model))
  best_model = os.path.join(model_dir, best_model)
  best_model = load_model(best_model)
    
  return best_model

# return 0 for single gpu, otherwise: n for multi gpus
def init_gpu(cuda_devices):
  multi_gpu = 0
  if cuda_devices == '-1':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(notebook_util.pick_gpu_lowest_memory()) 
  else:
    device_idx = int(cuda_devices)
    if device_idx >= 0:
      os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    else:
      multi_gpu = -device_idx
      gpus = notebook_util.pick_gpu_lowest_memory(multi_gpu)
      str_gpus = []
      for gpu in gpus:
        str_gpus.append(str(gpu))
      str_devices = ",".join(str_gpus)
      os.environ["CUDA_VISIBLE_DEVICES"] = str_devices
      print('multi gpu is enabled:', str_devices)
      print('multi_gpu:', multi_gpu)
  return multi_gpu

# get class names from dir or list
def get_class_names(list_mode, train_dir):
  class_names = set()
  if list_mode == 0:
    for subdir in sorted(os.listdir(FLAGS.train_dir)):
      if os.path.isdir(os.path.join(FLAGS.train_dir, subdir)):
        class_names.add(subdir)
  else:
    with open(FLAGS.train_dir, "r") as f:
      for line in f:
        vec = line.strip().split()
        if len(vec) != 2:
          print("line format error in file '" + FLAGS.train_dir+ "': " + line)
          sys.exit(0)
        label = vec[1]
        if label not in class_names:
          class_names.add(label)
  return class_names

# auto inference list_mode
def get_list_mode(train_dir, val_dir, test_dir):
  assert os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir)
  is_dir1 = os.path.isdir(train_dir)
  is_dir2 = os.path.isdir(val_dir)
  is_dir3 = os.path.isdir(test_dir)
  assert is_dir1 == is_dir2 and is_dir1 == is_dir3
  return 0 if is_dir1 else 1

# ensure len(params)==nstage
def get_stage_param_list(param, nstage, transf=int):
  param_list = list(map(transf, param.split(',')))
  if len(param_list) >= nstage:
    return param_list[: nstage]
  else:
    # repeat last element until len(params)==nstage
    return param_list + [param_list[-1]] * (nstage - len(param_list))

# main process
def main():
  # config stage params
  # typical we train newly added fc for 1st stage (a few epochs)
  # and then train all layers except starting cnn layers for 2nd stage
  epochs = list(map(int, FLAGS.epochs.split(',')))
  nstage = len(epochs)
  batch_size = get_stage_param_list(FLAGS.batch_size, nstage)
  optimizer = get_stage_param_list(FLAGS.optimizer, nstage, transf=str)
  finetune_blocks = get_stage_param_list(FLAGS.finetune_blocks, nstage)
  random_crop = get_stage_param_list(FLAGS.random_crop, nstage)
  data_augmentation = get_stage_param_list(FLAGS.data_augmentation, nstage)
  lr_all_stages = init_lr_list(FLAGS, epochs)
  assert nstage == len(batch_size) and nstage == len(optimizer) and nstage == \
    len(lr_all_stages) and nstage == len(finetune_blocks) and nstage == len(random_crop) and nstage == len(data_augmentation)
 
  # start main
  multi_gpu = init_gpu(FLAGS.cuda_devices)
  random_seed = 0
  
  # Define Distribution Strategy
  # https://www.tensorflow.org/tutorials/distribute/keras
  #strategy = tf.contrib.distribute.MirroredStrategy()
  #print ('Number of devices: {}'.format(strategy.num_towers)) 
 
  # inference class names
  list_mode = get_list_mode(FLAGS.train_dir, FLAGS.validation_dir, FLAGS.test_dir)
  if list_mode:
    print('list mode will be used for dataset')
    os.system('cp %s %s/' % (FLAGS.train_dir, FLAGS.output_dir))
    os.system('cp %s %s/' % (FLAGS.validation_dir, FLAGS.output_dir))
    os.system('cp %s %s/' % (FLAGS.test_dir, FLAGS.output_dir))
  else:
    print('directory mode will be used for dataset')
  class_names = get_class_names(list_mode, FLAGS.train_dir)
  assert class_names == get_class_names(list_mode, FLAGS.validation_dir)
  num_classes = len(class_names)
  print('found total of %d classes in dir: %s' % (num_classes, FLAGS.train_dir))
  
  # config model
  if multi_gpu:
    with tf.device('/cpu:0'):
      model, image_size, preprocess_input = set_model(FLAGS.model, num_classes,
          FLAGS.weight_decay, FLAGS.image_size, FLAGS.se_block, FLAGS.rgb_tsf_alpha, FLAGS.optimizing_type=="classify")
  else:
    model, image_size, preprocess_input = set_model(FLAGS.model, num_classes,
        FLAGS.weight_decay, FLAGS.image_size, FLAGS.se_block, FLAGS.rgb_tsf_alpha, FLAGS.optimizing_type=="classify")
  
  interpolation = FLAGS.interpolation
  letter_box = FLAGS.letter_box
 
  # generate train/validation/test sequence/generator
  print('generate validation data generator')
  validation_generator = get_generator(preprocess_input, FLAGS.validation_dir, list_mode,
      image_size, interpolation, letter_box, batch_size[0], random_crop[0], random_seed, FLAGS, FLAGS.aug_on_val != 0)
  print('generate train data generator')
  train_generator = get_generator(preprocess_input, FLAGS.train_dir, list_mode,
      image_size, interpolation, letter_box, batch_size[0], random_crop[0], random_seed, FLAGS, True, val_sequence=(None if data_augmentation[0] else validation_generator))
  print('generate test data generator')
  test_generator = get_generator(preprocess_input, FLAGS.test_dir, list_mode,
      image_size, interpolation, letter_box, batch_size[0], random_crop[0], random_seed, FLAGS, False)
  train_validation_files = len(train_generator.filenames) + len(validation_generator.filenames)
  print('class indices: %s' % train_generator.class_indices)

  # class weight
  class_weight = None
  if FLAGS.class_weight != '0':
    class_weight = eval(FLAGS.class_weight)
    assert isinstance(class_weight, dict)
    print('class weight used for weighting the loss function is ', FLAGS.class_weight)
  
  # history metrics
  history_metrics = []
  best_model_idx = -1

  # save the labels define file
  labels_file = os.path.join(FLAGS.output_dir, 'labels.txt') 
  print('save labels to: %s' % labels_file)
  save_labels_file(labels_file, train_generator.class_indices)

  # start training stages
  current_epoch = 0
  epoch_turn_log = []
  for stage in range(nstage):
    stage_lr = lr_all_stages[stage]
    stage_batch_size = batch_size[stage]
    stage_random_crop = random_crop[stage]
    stage_data_aug = data_augmentation[stage]
    initial_lr = stage_lr[0]

    print('*************************************************************')
    print('start stage %d for total %d epochs' % (stage+1, epochs[stage]))
    print('stage lrs are: %s' % stage_lr)
    print('stage data augmentation for train dataset: %s' % ('on' if stage_data_aug else 'off'))

    # load best model in the last stage
    if stage > 0:
      if multi_gpu:
        with tf.device('/cpu:0'):
          model = load_best_model(FLAGS.model, FLAGS.output_dir, stage-1, best_model_idx)
      else:
        model = load_best_model(FLAGS.model, FLAGS.output_dir, stage-1, best_model_idx)
      if FLAGS.io_pipeline == 'dataset':
        print('generate validation data generator')
        validation_generator = get_generator(preprocess_input, FLAGS.validation_dir, list_mode,
                                             image_size, interpolation, letter_box, batch_size[stage], random_crop[stage], random_seed, FLAGS, FLAGS.aug_on_val != 0)
        print('generate train data generator')
        train_generator = get_generator(preprocess_input, FLAGS.train_dir, list_mode,
                                        image_size, interpolation, letter_box, batch_size[stage], random_crop[stage], random_seed, FLAGS, True, val_sequence=(None if data_augmentation[stage] else validation_generator))
        print('generate test data generator')
        test_generator = get_generator(preprocess_input, FLAGS.test_dir, list_mode,
                                       image_size, interpolation, letter_box, batch_size[stage], random_crop[stage], random_seed, FLAGS, False)

    # config trainable
    print('config trainable layers')
    set_trainable(finetune_blocks[stage], model, FLAGS.quickly_ft)
    
    # update batch_size and random_crop
    print('update train generator: batch_size=%d random_crop=%s' %
        (stage_batch_size, str(stage_random_crop)))
    if FLAGS.io_pipeline == 'sequence':
      train_generator.update_param(stage_batch_size, stage_random_crop, val_sequence=(None if stage_data_aug else validation_generator))
      print('update validation generator: batch_size=%d random_crop=%s' %
        (stage_batch_size, str(stage_random_crop)))
      if FLAGS.aug_on_val != 0:
        validation_generator.update_param(stage_batch_size, stage_random_crop, val_sequence=(None if stage_data_aug else validation_generator))
    # 测试数据不需要更新batch_szie和random_crop参数
    #test_generator.update_param(stage_batch_size, 0)
    steps_train, steps_validation, _ = get_generator_steps(train_generator,
        validation_generator, test_generator, FLAGS)
    
    # config optimizer
    print('config optimizer: %s %s' % (optimizer[stage], initial_lr))
    opt = set_optimizer(optimizer[stage], initial_lr, FLAGS.clipnorm)
    workers = 16 if multi_gpu else 8
    
    # config callbacks
    print('config callbacks')
    early_stopping_by_val_acc = FLAGS.early_stopping_by_val_acc
    callbacks = set_callbacks(train_validation_files,
        FLAGS, train_generator, test_generator, validation_generator, stage_lr, stage,
        history_metrics, workers)
    
    # compile the model (should be done *after* setting layers to non-trainable)
    print('model compile')
    if multi_gpu:
      final_model = multi_gpu_model(model, gpus=multi_gpu)
      if FLAGS.optimizing_type == "classify":
        final_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
      else:
        final_model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    else:
      final_model = model
      if FLAGS.optimizing_type == "classify":
        final_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
      else:
        final_model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    #if FLAGS.optimizing_type == "classify":
    #  final_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #else:
    #  final_model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    
    print('start training')
    if FLAGS.io_pipeline == 'sequence':
      final_model.fit_generator(
        train_generator,
        #int(steps_train/FLAGS.epoch_scale),
        epochs=epochs[stage],
        validation_data=validation_generator,
        #validation_steps=int(steps_validation/FLAGS.epoch_scale),
        callbacks=callbacks,
        use_multiprocessing=FLAGS.multiprocessing,
        workers=workers,
        class_weight=class_weight,
        max_queue_size=32)
    else:
      final_model.fit(
        train_generator,
        steps_per_epoch = int(steps_train),
        epochs=epochs[stage],
        validation_data=validation_generator,
        validation_steps=int(steps_validation),
        callbacks=callbacks,
        use_multiprocessing=FLAGS.multiprocessing,
        workers=workers,
        class_weight=class_weight,
        max_queue_size=32)
    
    for callback in callbacks:
      if isinstance(callback, RocCallback):
        history_metrics.append(callback.metrics)
        best_model_idx = callback.best_model_idx
        break

    # deallocate gpu memory
    K.clear_session()
        
def add_argument(parser, opt):
  if opt == 'preprocess':
    parser.add_argument('--data_augmentation', type=str, default='1',
      help='Whether to use data augmentation for train(random crop is independently controled by --random_crop), and it could be "1,1,1,0".')
    parser.add_argument('--horizontal_flip', type=int, default=0,
      help='Whether to use preprocess horizontal flip.')
    parser.add_argument('--vertical_flip', type=int, default=0,
      help='Whether to use preprocess vertical_flip.')
    parser.add_argument('--rotation_range', type=int, default=0,
      help='Degree range for preprocess random rotations.')
    parser.add_argument('--rotation90', type=int, default=0,
      help='Whether to use preprocess random generate 0, 90, 180 or 270 degrees variant.')
    parser.add_argument('--brightness', type=float, default=0.0,
      help='Whether to use preprocess random brightness, final range is [1-x, 1+x].')
    parser.add_argument('--saturation', type=float, default=0.0,
      help='Whether to use preprocess random saturation, final range is [1-x, 1+x].')
    parser.add_argument('--zoom_pyramid_levels', type=int, default=1,
      help='Whether to use preprocess multiscale training, image pyramid of given levels will be built.')
    parser.add_argument('--zoom_short_side', type=int, default=0,
      help='The short side of the image in the top layer of the image pyramid when multiscale training is enabled.')
    parser.add_argument('--gaussian_blur', type=int, default=0,
      help='Whether using gaussian blur on images.')
    parser.add_argument('--motion_blur', type=float, default=0.,
      help='The probability to use motion blur, and the range is [0.0, 1.0].')
    # argument of random crop
    parser.add_argument('--random_crop', type=str, default='0',
      help='Whether to use random crop images for train.')
    parser.add_argument('--min_object_covered', type=float, default=1.0,
      help='An optional `float`. Defaults to `1.0`. The cropped area of the image must contain at least this fraction of any bounding box supplied. This parameter is not working for currently.')
    parser.add_argument('--aspect_ratio_range', type=str, default='0.75,1.33',
      help='An optional list of `floats`. Defaults to `0.75,1.33` The cropped area of the image must have an aspect ratio = width / height within this range.')
    parser.add_argument('--area_range', type=str, default='0.05,1.0',
      help='An optional list of `floats`. Defaults to `0.05,1.0` The cropped area of the image must contain a fraction of the supplied image within in this range.')
    parser.add_argument('--random_crop_list', type=str, default='',
      help='The path of list file with md5 (with bounding box info or not) of imgs which would be random cropped.')
    parser.add_argument('--class_aware_sampling', type=int, default=0,
      help='Whether class_aware sampling on train set.')
    parser.add_argument('--save_to_dir', type=str, default='augmented_imgs',                        
      help='This allows you to optionally specify a directory to which to save the augmented pictures being generated')
    parser.add_argument('--save_prefix', type=str, default='aug',
      help='Prefix to use for filenames of saved pictures')
    parser.add_argument('--save_samples_num', type=int, default=0,
      help='The num of augmented pictures will be sampled and saved.')
    parser.add_argument('--interpolation', type=str, default='random',
      help='Interpolation method used to resample the image if the target size is different from that of the loaded image, e.g:bilinear, bicubic, lanczos, antialias. if input "random", it will pick interpolation method randomly')
    parser.add_argument('--letter_box', type=int, default=0,
      help='Scale with keeping aspect ratio when letter box is opened, padding black on margin.')
    parser.add_argument('--mixup', type=int, default=1,
      help='Form a new example by a weighted linear interpolation of two examples which randomly sampled.')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
      help='Mixup interpolation coefficient.')
    parser.add_argument('--io_pipeline', type=str, default='sequence',
      help='Which io pipeline to use, sequence/dataset')
  elif opt == 'dataset':
    parser.add_argument('--train_dir', type=str, default='data/train',
      help='Path to a folder for train labeled images.')
    parser.add_argument('--validation_dir', type=str, default='data/validation',
      help='Path to a folder for validation labeled images.')
    parser.add_argument('--test_dir', type=str, default='data/test',
      help='Path to a folder for test labeled images.')
  elif opt == 'lr':
    parser.add_argument('--learning_rate', type=str, default="0.001",
      help='How large a learning rate to use when training.')
    # lr decay
    parser.add_argument('--lr_decay', type=int, default=0,
      help='Whether turn on the exponential learning rate decay.')
    parser.add_argument('--lr_decay_decayrate', type = float, default=0.94,
      help='The decay rate when --lr_decay flag is turned on.')
    parser.add_argument('--lr_decay_decaystep', type = int, default=2,
      help='Every n epoches the learning rate is decayed.')
  elif opt == 'common':
    parser.add_argument('--batch_size', type=str, default='100',
      help='How many images to train/validation/test on at a time.')
    parser.add_argument('--epochs', type=str, default='4',
      help='How many training epochs to run before ending.')
    #parser.add_argument('--epoch_scale', type=int, default=1,
    #  help='Quick to finisn an epoch only for testing purpos, the result is not correct')
    parser.add_argument('--cuda_devices', type=str, default="-1",
      help='Which gpus to use when training, use -1 to auto select a gpu with lowest memory, use -n to start multi-gpu training.')
    parser.add_argument('--output_dir', type=str, default='.',
      help='Where to save the trained checkpoints.')
    parser.add_argument('--validation_auc', type=int, default=0,
      help='Whether do the calculation of validation auc.')
    parser.add_argument('--test_auc', type=int, default=0,
      help='Whether do the calculation of test auc, should be set to 0 for the new sequence version.')
    parser.add_argument('--clipnorm', type=float, default=0.,
      help='All parameter gradients will be clipped to this specified maximum norm.')
    parser.add_argument('--optimizer', type=str, default='adam',
      help='Which optimizer to use, support adam, rmsprop and sgd.')
    parser.add_argument('--early_stopping', type = int, default=5,
      help='Number of epochs with no improvement after which training will be stopped, 0 means early_stopping is off.')
    parser.add_argument('--finetune_blocks', type=str, default='0',
      help='Which block to start finetune, options for inception_v3 is [0,10], for xception is [0,14], for mobilenet_1.00_224 is [0,14]')
    parser.add_argument('--multiprocessing', type=int, default=0,
      help='Whether turn on the multiprocessing optimization.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
      help='Whether turn on label smoothing, the epsilon param for LS.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
      help='The weight decay on the model weights.')
    parser.add_argument('--optimizing_type', type=str, default="classify",
      help='Optimizing type is "classify" or "regression".')
    parser.add_argument('--class_indices', type=str, default="",
      help='This argument takes classes string sorted by correlation, exp "normal,sexy,porn" , and it`s open when optimizing type is "regression".')
    parser.add_argument('--classes_map', type=str, default="",
      help='Input a list file to realize classes num reduction, e.g. "porn_6classes_to_4classes.list"')
    parser.add_argument('--pooling', type=str, default='avg',
      help='Choose which Pooling before Dense, e.g: avg, max, max-avg')
    parser.add_argument('--rgb_tsf_alpha', type=float, default=0.0,
      help='Whether to use mini-networks for learned colorspace transformations, which after image preprocessing. If value is not 0.0, that it must be 0.1 or 0.5 as alpha in VLReLU.')
    parser.add_argument('--class_weight', type=str, default='0',
      help='Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class. e.g.{0:1, 1:10}, default value "0" means that class_weight is None.')
    parser.add_argument('--aug_on_val', type=int, default=0,
      help='Whether turn on data augmentation on validation set.')
    parser.add_argument('--early_stopping_by_val_acc', type=float, default=0,
      help='Whether fine-tune on a badcase_dataset to cover these bad cases.')

# argument
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_argument(parser, 'dataset')
  add_argument(parser, 'common')
  add_argument(parser, 'preprocess')
  add_argument(parser, 'lr')
  parser.add_argument('--dense_units', type=int, default=1024,
    help='The units of the 1st full connect layer right after the top of pretrained model, 0 means 1st fc is not used.')
  parser.add_argument('--model', type=str, default='inception_v3',
    help='Which model we will use, default is inception_v3, you can use inception_v3, resnet50, xception, mobilenet_1.00_224, inception_resnet_v1, inception_resnet_v2, densenet201, nasnet_large, resnet50_v2, resnext50.')
  parser.add_argument('--image_size', type=int, default=0,
    help='Define the image_size of the model.')
  parser.add_argument('--se_block', type=int, default=0,
    help='The block of se, 0 means seblock is not used.')
  parser.add_argument('--quickly_ft', type=int, default=0,
    help='Train with quick mode, just train softmax layer, 0 means normal train mode.')
  FLAGS, unparsed = parser.parse_known_args()
  print(FLAGS)
  if len(unparsed) > 0:
    print("there are unknow args %s " % ','.join(unparsed))
    sys.exit(-1)
  main()
