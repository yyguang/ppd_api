#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <cao-teng-teng.cao@foxmail.com> 2017-11-08 15:34:06

import numpy as np
import math
import sys
import os
import random
# tf.keras.preprocessing was avaliable on tensorflow 1.13
from keras_preprocessing.image import flip_axis, array_to_img
from keras_preprocessing.image.utils import _list_valid_filenames_in_directory, img_to_array
from keras_preprocessing.image.affine_transformations import apply_affine_transform

from tensorflow.python.keras.utils.data_utils import Sequence
import multiprocessing.pool
from tensorflow.keras import backend as K
from copy import deepcopy
try:
  import queue
except ImportError:
  import Queue as queue
import threading
import cv2
import imghdr

try:
  from PIL import ImageEnhance, ImageFilter, Image as pil_image
except ImportError:
  print('error import PIL')
  pil_image = None
  ImageEnhance = None

if pil_image is not None:
  _PIL_INTERPOLATION_METHODS = {
    'nearest': pil_image.NEAREST,
    'bilinear': pil_image.BILINEAR,
    'bicubic': pil_image.BICUBIC,
    'antialias': pil_image.ANTIALIAS
  }
  # These methods were only introduced in version 3.4.0 (2016).
  if hasattr(pil_image, 'HAMMING'):
    _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
  if hasattr(pil_image, 'BOX'):
    _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
  # This method is new in version 1.1.3 (2013).
  if hasattr(pil_image, 'LANCZOS'):
    _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS

# lock for func supplement_queue
mutex_supplement_queue = threading.Lock()

def get_img_size(img):
  if isinstance(img, np.ndarray):
    h, w = img.shape[:2]
  else:
    w, h = img.size[0], img.size[1]
  return (w, h)

def read_random_crop_dict(random_crop_file_path):
  random_crop_dict = {}
  num_img_full = 0
  num_img_rect = 0
  if os.path.exists(random_crop_file_path):
    with open(random_crop_file_path, 'r') as random_crop_file:
      for line in random_crop_file.readlines():
        if "\t" not in line:
          random_crop_dict[line.strip('\n')] = []
          num_img_full += 1
        else:
          num_img_rect += 1
          if len(line.strip().split("\t")) != 2:
            print('error format in random crop, line: %s' % line)
          img, coords = line.strip().split("\t")
          coords = coords.split(",")
          if len(coords) % 4 != 0:
            print('error format in random crop, line: %s' % line)
            continue
          bbox_data = []
          data = []
          for i in range(len(coords)):
            data.append(float(coords[i]))
            i += 1
            if i % 4 == 0:
              bbox_data.append(data)
              data = []
          random_crop_dict[img] = bbox_data
    print('random crop is enabled with file %s, num_img_full: %d, num_img_rect: %d' 
        % (random_crop_file_path, num_img_full, num_img_rect))
  return random_crop_dict 

# Notes: min_object_covered is not working currently
def random_crop_img(img_PIL, bbox_data, min_object_covered, aspect_ratio_range, area_range, attempts=100):
  w, h = get_img_size(img_PIL) 
  if len(bbox_data) == 0:
    bbox = [0.0, 0.0, 1.0, 1.0]
  else:
    try:
      bbox = []
      for d in bbox_data:
        d_new = []
        d_new.append(d[1]/h if d[1] > 0.0 else 0.0)
        d_new.append(d[0]/w if d[0] > 0.0 else 0.0)
        d_new.append(d[3]/h if d[3] < h else 1.0)
        d_new.append(d[2]/w if d[2] < w else 1.0)
        bbox.append(d_new)
      bbox = bbox[random.randint(0, len(bbox)-1)]
      min_object_covered = 1.0
    except Exception as e:
      print('exception random crop img:',  e)
      bbox = [0.0, 0.0, 1.0, 1.0]
      min_object_covered = 0.5
  # 带bounding box参数的crop
  if bbox != [0.0, 0.0, 1.0, 1.0]:
    min_area_range = max((bbox[3]-bbox[1])*(bbox[2]-bbox[0]),area_range[0])
    if min_area_range >= area_range[1]:
      return img_PIL, False

    # 计算box距离左下右上哪个角更近
    is_near_left_down = True
    if (1.-bbox[3])*(1.-bbox[2]) < bbox[1]*bbox[0]:
      is_near_left_down = False
    for i in range(attempts):
      r_aspect_ratio = random.random() * (aspect_ratio_range[1]-aspect_ratio_range[0]) + aspect_ratio_range[0]
      r_area_range = random.random() * (area_range[1]-min_area_range) + min_area_range
      r_height = math.sqrt(r_area_range*w*h/r_aspect_ratio)
      r_width = r_height*r_aspect_ratio
      if r_width > w or r_height > h:
        continue
      if is_near_left_down:
        r_xmin = random.random()*(bbox[1]-max(0.0,bbox[1]-r_width/w))+max(0.0,bbox[1]-r_width/w)
        r_ymin = random.random()*(bbox[0]-max(0.0,bbox[0]-r_height/h))+max(0.0,bbox[0]-r_height/h)
        xmax = r_xmin + r_width/w
        ymax = r_ymin + r_height/h
        if bbox[2]<=ymax<=1. and bbox[3]<=xmax<=1.:
          box = (int(r_xmin*w), int(r_ymin*h), int(xmax*w), int(ymax*h))
          assert (box[2] - box[0]) * (box[3] - box[1]) >= area_range[0]
          if isinstance(img_PIL, np.ndarray):
            return img_PIL[box[1]:box[3], box[0]:box[2]], True
          else:
            return img_PIL.crop(box), True
        else:
          continue
      else:
        r_xmax = random.random()*(min(1.0,bbox[3]+r_width/w)-bbox[3])+bbox[3]
        r_ymax = random.random()*(min(1.0,bbox[2]+r_height/h)-bbox[2])+bbox[2]
        xmin = r_xmax - r_width/w
        ymin = r_ymax - r_height/h
        if 0.<=ymin<=bbox[0] and 0.<=xmin<=bbox[1]:
          box = (int(xmin*w), int(ymin*h), int(r_xmax*w), int(r_ymax*h))
          assert (box[2] - box[0]) * (box[3] - box[1]) >= area_range[0]
          if isinstance(img_PIL, np.ndarray):
            return img_PIL[box[1]:box[3], box[0]:box[2]], True
          else:
            return img_PIL.crop(box), True
        else:
          continue
    else:
      return img_PIL, False
  # 全图的random crop
  else:
    min_area_range = area_range[0]
    for i in range(attempts):
      r_aspect_ratio = random.random() * (aspect_ratio_range[1]-aspect_ratio_range[0]) + aspect_ratio_range[0]
      r_area_range = random.random() * (area_range[1]-min_area_range) + min_area_range
      r_height = math.sqrt(r_area_range*w*h/r_aspect_ratio)
      r_width = r_height*r_aspect_ratio
      if r_width > w or r_height > h:
        continue
      r_xmin = random.random() * (1.0-r_width/w)
      r_ymin = random.random() * (1.0-r_height/h)
      xmax = r_xmin+r_width/w
      ymax = r_ymin+r_height/h
      box = (int(r_xmin*w), int(r_ymin*h), int(xmax*w), int(ymax*h))
      assert (box[2] - box[0]) * (box[3] - box[1]) >= area_range[0]
      if isinstance(img_PIL, np.ndarray):
        return img_PIL[box[1]:box[3], box[0]:box[2]], True
      else:
        return img_PIL.crop(box), True
    else:
      return img_PIL, False


# 针对高斯模糊的radius进行固定概率取值，各值概率比例如“rate”所示
def get_random_radius():
  rate = [32, 16, 4, 1]
  radius_list = [0, 1, 2, 3]
  start = 0
  randnum = random.randint(1, sum(rate))
  for index, item in enumerate(rate):
    start += item
    if randnum <= start:
      break
  return radius_list[index]

# motion blur 运动模糊
# degree越大则模糊程度越高，angle表示运动模糊kernel矩阵的角度
def motion_blur(image, degree=10, angle=20):
  image = np.asarray(image)
  # 生成任意角度的运动模糊kernel的矩阵，degree越大，模糊程度越高
  M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
  motion_blur_kernel = np.diag(np.ones(degree))
  motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

  motion_blur_kernel = motion_blur_kernel/degree
  blurred = cv2.filter2D(image, -1, motion_blur_kernel)

  cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
  blurred = np.array(blurred, dtype=np.uint8)
  image_mb = pil_image.fromarray(blurred)
  return image_mb

# 当brightness和saturation都开启时，两种先后顺序进行随机
def distort_color(img, brightness, saturation):
  if brightness != 0.0 and saturation == 0.0:
    ImageEnhance_Brightness = ImageEnhance.Brightness(img)
    u = np.random.uniform(1-brightness, 1+brightness)
    img = ImageEnhance_Brightness.enhance(u)
  elif brightness == 0.0 and saturation != 0.0:
    ImageEnhance_Color = ImageEnhance.Color(img)
    u = np.random.uniform(1-saturation, 1+saturation)
    img = ImageEnhance_Color.enhance(u)
  elif brightness != 0.0 and saturation != 0.0:
    if np.random.random() < 0.5:
      ImageEnhance_Brightness = ImageEnhance.Brightness(img)
      u = np.random.uniform(1-brightness, 1+brightness)
      img = ImageEnhance_Brightness.enhance(u)
      ImageEnhance_Color = ImageEnhance.Color(img)
      u = np.random.uniform(1-saturation, 1+saturation)
      img = ImageEnhance_Color.enhance(u)
    else:
      ImageEnhance_Color = ImageEnhance.Color(img)
      u = np.random.uniform(1-saturation, 1+saturation)
      img = ImageEnhance_Color.enhance(u)
      ImageEnhance_Brightness = ImageEnhance.Brightness(img)
      u = np.random.uniform(1-brightness, 1+brightness)
      img = ImageEnhance_Brightness.enhance(u)
  return img

# ref:
# https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/image_data_generator.py#L742
def random_transform(x, data_format, rotation_range, rotation90, horizontal_flip, vertical_flip):
  """Randomly augment a single image tensor.
  # Arguments
    x: 3D tensor, single image.
  # Returns
    A randomly transformed version of the input (same shape).
  """
  if data_format == 'channels_first':
    channel_axis = 1
    row_axis = 2
    col_axis = 3
  if data_format == 'channels_last':
    channel_axis = 3
    row_axis = 1
    col_axis = 2
  
  # x is a single image, so it doesn't have image number at index 0
  img_row_axis = row_axis - 1
  img_col_axis = col_axis - 1
  img_channel_axis = channel_axis - 1
 
  theta = 0 
  if rotation_range:
    rotation_n = np.random.random_integers(0, rotation_range)
    theta = rotation_n

  if rotation90:
    rotation_n = np.random.random_integers(0, 3)
    theta = rotation_n * 90

  tx, ty = 0, 0
  shear = 0
  zx, zy = 1, 1 
  x = apply_affine_transform(x, theta, tx, ty, shear, zx, zy,
      row_axis=img_row_axis,
      col_axis=img_col_axis,
      channel_axis=img_channel_axis)

  if horizontal_flip == 1:
    if np.random.random() < 0.5:
      x = flip_axis(x, img_col_axis)

  if vertical_flip:
    if np.random.random() < 0.5:
      x = flip_axis(x, img_row_axis)

  return x


# Read a image form file and do augmentation, return ndarry format
# this function is taken original from keras, and fix for the error of:
# Palette images with Transparency expressed in bytes converted to RGBA images
def sm_load_img(path, data_format, rotation_range=0, rotation90=0, horizontal_flip=0,
    vertical_flip=0, grayscale=False, target_size=None, interpolation='random', letter_box=0, levels=1,
    short_side=0, hflip=False, random_crop=False, bbox_data=[],
    min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33), area_range=(0.05,
      1.0), gaussian_blur=0, motion_blur=0, brightness=0.0, saturation=0.0, process_lib=0):
  """Loads an image into PIL format.

  # Arguments
      path: Path to image file
      random_crop: Boolean, whether to random crop.
      grayscale: Boolean, whether to load the image as grayscale.
      target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
      interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
      letter_box: 将原图保持宽高比缩放，恰好放置于target size的底片中央，
            空白部分用黑色填充.
      process_lib: 
            0: pil
            1: opencv

  # Returns
      A PIL Image instance.

  # Raises
      ImportError: if PIL is not available.
      ValueError: if interpolation method is not supported.
  """
  # load image
  if process_lib == 0:
    if pil_image is None:
      raise ImportError('Could not import PIL.Image. '
                      'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
      if img.mode != 'L':
        img = img.convert('L')
    else:
      if img.mode in ('L', 'RGB', 'P') and "transparency" in img.info and \
        img.info['transparency'] is not None:
        t = img.info['transparency']
        if isinstance(t, bytes):
          img = img.convert('RGBA')
      if img.mode != 'RGB':
        img = img.convert('RGB')
  else:
    if grayscale:
      img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
      img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
      print('failed cv2.imread:', path)

  result = False
  if random_crop:
    try:
      if len(bbox_data) == 0:
        img, result = random_crop_img(img, bbox_data, min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range, area_range=area_range)
      else:
        img, result = random_crop_img(img, bbox_data, min_object_covered=1.0,
            aspect_ratio_range=aspect_ratio_range, area_range=area_range)
    except Exception as e:
      print('exception random_crop:', e)
 
  # Gaussian Blur, todo: add support with opencv
  assert gaussian_blur == 0
  if gaussian_blur:
    blur_radius = get_random_radius()
    if blur_radius:
      img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

  # Motion blur, todo: add supoort with opencv
  assert motion_blur == 0
  if motion_blur:
    if random.random() <= motion_blur:
      img = motion_blur(img)
 
  # resize所使用的resample方法
  resample = 0 # 给定默认值，对应NEAREST
  if interpolation == 'random':
    methods = list(_PIL_INTERPOLATION_METHODS.keys())
    random.shuffle(methods)
    resample = _PIL_INTERPOLATION_METHODS[methods[0]]
  elif interpolation not in _PIL_INTERPOLATION_METHODS:
    raise ValueError(
      'Invalid interpolation method {} specified. Supported '
      'methods are {}'.format(
      interpolation,
      ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
  else:
    resample = _PIL_INTERPOLATION_METHODS[interpolation]

  # multiscale：确定所有scale的列表，产生随机数，确定最终scale
  if levels == 1 or short_side == 0:
    pass
  else:
    w, h = get_img_size(img)
    max_scale = min(w, h) / float(short_side)
    basic_scale = pow(max_scale, 1.0 / (levels-1))
    scales = [pow(basic_scale, i) for i in range(levels)]
    random_idx = np.random.random_integers(0, levels-1)
    final_scale = scales[random_idx]
    nw, nh = round(w/final_scale), round(h/final_scale)
    nw = max(nw, 1) # 防止为0
    nh = max(nh, 1)
    width_height_tuple = (int(nw), int(nh))
    #print('scales: %s, size: %s -> %s' % (scales, img.size, hw_tuple))
    if (w, h) != width_height_tuple:
      if process_lib == 0:
        img = img.resize(width_height_tuple, resample)
      else:
        img = cv2.resize(img, width_height_tuple)
  
  # resize
  if target_size:
    width_height_tuple = (target_size[1], target_size[0])
    w, h = get_img_size(img)
    if (w, h) != width_height_tuple:
      # letter box 逻辑实现, todo: add support with opencv
      assert letter_box == 0
      if letter_box and target_size[0] == target_size[1]:
        back_groud_color = (0, 0, 0)
        back_groud = pil_image.new('RGB', width_height_tuple, back_groud_color)
        if max(w, h) != target_size[0]:
          if w > h:
            nw = target_size[0]
            nh = h * target_size[0]/w
          else:
            nh = target_size[0]
            nw = w * target_size[0]/h
        else:
          nw, nh = w, h
        nw, nh = round(nw), round(nh)
        width_height_tuple = (int(nw), int(nh))
        img = img.resize(width_height_tuple, resample)
        back_groud.paste(img, (int(round(target_size[0]/2)-round(nw/2)), int(round(target_size[0]/2)-round(nh/2))))
        img = back_groud
      else:
        if process_lib == 0:
          img = img.resize(width_height_tuple, resample)
        else:
          img = cv2.resize(img, width_height_tuple)
  
  # hflip
  if hflip:
    if process_lib == 0:
      img = img.transpose(pil_image.FLIP_LEFT_RIGHT)
    else:
      img = cv2.flip(image, 1)
    
  # color distort, todo: add support with opencv
  if brightness != 0.0 or saturation != 0.0:
    if process_lib == 0:
      img = distort_color(img, brightness, saturation)

  # convert to ndarray
  if isinstance(img, np.ndarray):
    x = img
  else:
    x = img_to_array(img, data_format=data_format)
      
  # random transform
  if (horizontal_flip == 1) or vertical_flip or rotation90 or rotation_range:
    x = random_transform(x, data_format, rotation_range, rotation90, horizontal_flip, vertical_flip)
      
  return x, result

# multiscale implement:
# an image pyramid is constructed with zoom_pyramid_levels levels
# the short side of the image for the 1st level of pyramid is specified by zoom_short_side
class SMSequence(Sequence):
  def __init__(self, directory, batch_size, image_size, preprocessing_function, optimizing_type='classify', class_indices="",
      interpolation='random', letter_box=0, val_sequence=None,
      rotation90=0, rotation_range=0, horizontal_flip=0, vertical_flip=0, random_seed=0,
      shuffle=True, zoom_pyramid_levels=1, zoom_short_side=0,
      brightness=0.0, saturation=0.0,
      random_crop=0, min_object_covered=0.1,
      aspect_ratio_range=(0.75, 1.33),
      area_range=(0.05, 1.0), random_crop_list="",
      gaussian_blur=0,
      motion_blur=0,
      label_smoothing=0.0,
      class_aware_sampling=0, min_queue_len=100, list_mode=0,
      save_to_dir='augmented_imgs', save_prefix='aug', save_samples_num=0, mixup=0, mixup_alpha=0.2):
    '''
    # Arguments
      rotation90: Boolean. Randomly rotate 0, 90, 180 or 270 degrees.
      rotation_range: Float. Randomly rotate an angle by degrees.
      horizontal_flip: Int. 1: Randomly flip inputs horizontally, 2: Double the inputs with its horizontally flip variant.
      vertical_flip: Boolean. Randomly flip inputs vertically.
    '''

    self.optimizing_type = optimizing_type
    self.list_mode = list_mode
    self.class_aware_sampling = class_aware_sampling
    self.label_smoothing = label_smoothing
    self.random_seed = random_seed
    self.shuffle = shuffle
    # whether use same data augmentation on train dataset as on val dataset
    # val_sequence to get params of val generator
    self.val_sequence = val_sequence
    # save data augmentation
    self.save_to_dir = save_to_dir
    self.save_prefix = save_prefix
    self.save_samples_num = save_samples_num
    # mixup and mixup interpolation coefficient
    self.mixup = mixup
    self.mixup_alpha = mixup_alpha
    self.process_lib = 0

    self.zoom_pyramid_levels = zoom_pyramid_levels
    self.zoom_short_side = zoom_short_side
    self.brightness = brightness
    self.saturation = saturation
    self.horizontal_flip = horizontal_flip
    self.vertical_flip = vertical_flip
    self.rotation90 = rotation90
    self.rotation_range = rotation_range
    if rotation_range:
      print("Not support rotation_range now!")
      sys.exit(1)
    #self.do_random_transform = False
    #if (horizontal_flip == 1) or vertical_flip or rotation90 or rotation_range:
    #  self.do_random_transform = True
    self.directory = directory
    self.image_size = image_size
    self.interpolation = interpolation
    self.letter_box = letter_box
    self.preprocessing_function = preprocessing_function
    self.target_size = tuple((image_size, image_size))
    self.data_format = K.image_data_format()
    if self.data_format == 'channels_last':
      self.image_shape = self.target_size + (3,)
    else:
      self.image_shape = (3,) + self.target_size
    
    #if self.data_format == 'channels_first':
    #  self.channel_axis = 1
    #  self.row_axis = 2
    #  self.col_axis = 3
    #if self.data_format == 'channels_last':
    #  self.channel_axis = 3
    #  self.row_axis = 1
    #  self.col_axis = 2

    # random crop
    self.min_object_covered = min_object_covered
    self.aspect_ratio_range = aspect_ratio_range
    self.area_range = area_range
    self.random_crop_list = random_crop_list
    # 统计每个Epoch random crop的成功率
    self.random_crop_count = 0
    self.random_crop_success_count = 0.
   
    # Gaussian Blur
    self.gaussian_blur = gaussian_blur
    # Motion Blur
    self.motion_blur = motion_blur
 
    classes = []
    self.filenames = []
    if self.list_mode == 0:
      follow_links = False
      white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}
      # first, count the number of samples and classes
      for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
          classes.append(subdir)
      self.num_class = len(classes)
      self.class_indices = dict(zip(classes, range(len(classes))))
      if optimizing_type == "regression":
        self.class_indices = dict(zip(class_indices.split(","), range(len(classes))))
      pool = multiprocessing.pool.ThreadPool()

      results = []
      i = 0
      for dirpath in (os.path.join(directory, subdir) for subdir in classes):
        results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                            (dirpath, white_list_formats,(0.0,1.0),
                                             self.class_indices, follow_links)))
      classes_list = []
      for res in results:
        classes, filenames = res.get()
        classes_list.append(classes)
        self.filenames += filenames
      samples = len(self.filenames)
      self.classes = np.zeros((samples,), dtype='int32')
      for classes in classes_list:
        self.classes[i:i + len(classes)] = classes
        i += len(classes)
      pool.close()
      pool.join()
      self.classes = self.classes.tolist()	 
    else:
      # first, count the number of samples and classes
      labels = []
      count_not_exist = 0
      count_gif = 0
      with open(self.directory, "r") as f:
        for line in f:
          vec = line.strip().split()
          if len(vec) != 2:
            print("line format error in file '" + self.directory + "': " + line)
            sys.exit(0)
          label = vec[1]
          if not os.path.exists(vec[0]):
            print('file not exist in list: %s, label: %s' % (vec[0], label))
            count_not_exist += 1
            continue
          if imghdr.what(vec[0]) == 'gif':
            print('file gif found in list: %s, label: %s' % (vec[0], label))
            count_gif += 1
            continue
          labels.append(label)
          self.filenames.append(vec[0])
          if label not in classes:
            classes.append(label)
      classes = sorted(classes)
      self.num_class = len(classes)
      self.class_indices = dict(zip(classes, range(len(classes))))
      print('Count of files not exist: %d' % count_not_exist)
      print('Count of gif files: %d' % count_gif)

      # second, build an index of the images in the different class subfolders
      self.classes = [self.class_indices[label] for label in labels]
    print('Found %d images belonging to %d classes, detailed: ' % (len(self.filenames), self.num_class))
    
    label_count_dict = {}
    for class_index in self.classes:
      if class_index not in label_count_dict:
        label_count_dict[class_index] = 0
      label_count_dict[class_index] += 1
    print(label_count_dict)
    
    # 开启class-aware sampling时，构造一个数组filenames_cas，
    # 包含num_class项，每一项是一个队列Q
    if self.class_aware_sampling:
      self.min_queue_len = min_queue_len
      # init filenames_cas
      self.filenames_cas = []
      for i in range(self.num_class):
        self.filenames_cas.append(Queue.Queue(maxsize = 0))
      # compute sample_count
      self.sample_count = [0 for i in range(self.num_class)]
      for class_idx in self.classes:
        self.sample_count[class_idx] += 1
      # supplement all queue
      for class_idx in range(self.num_class):
        self.supplement_queue(class_idx)

    # update param
    self.update_param(batch_size, random_crop, val_sequence=self.val_sequence)
   
    # shuffle the data
    self.shuffle_data()
    
    # 采样需要保存数据增强结果的图片名
    if self.save_samples_num:
      if 0 < self.save_samples_num <= len(self.filenames): 
        self.save_data_augmentation_filenames=set(random.sample(self.filenames, self.save_samples_num))
      else:
        print('The save_samples_num is bigger than dataset, so there is to save all the augmented pictures being generated.')
        self.save_data_augmentation_filenames=set(self.filenames)
  
  # 更新参数，包括batch_size和random_crop
  def update_param(self, batch_size, random_crop, val_sequence=None):
    self.batch_size = batch_size
    self.random_crop = random_crop
    self.random_crop_dict = {}
    self.val_sequence = val_sequence
    if self.random_crop:
      if self.random_crop_list:
        random_crop_filepath = os.path.join(os.path.dirname(self.directory), self.random_crop_list)
        self.random_crop_dict = read_random_crop_dict(random_crop_filepath)
        if not self.random_crop_dict:
          print('Error: %s is empty or not exists' %(random_crop_filepath))
    
    len_ori = len(self.filenames)
    self.hflips = [False] * len_ori
    self.crops = [False] * len_ori
    if self.horizontal_flip == 2:
      tmp_filenames = deepcopy(self.filenames)
      tmp_classes = deepcopy(self.classes)
      self.filenames.extend(tmp_filenames)
      self.classes.extend(tmp_classes)
      self.hflips.extend([True] * len_ori)
      self.crops.extend([False] * len_ori)
      print('Double image with hflip.')
    if random_crop == 2:
      tmp_filenames = deepcopy(self.filenames)
      tmp_classes = deepcopy(self.classes)
      self.filenames.extend(tmp_filenames)
      self.classes.extend(tmp_classes)
      self.hflips.extend([False] * len_ori)
      self.crops.extend([True] * len_ori)
      print('Double image with random crop.')

      
  def shuffle_data(self):
    # random shuffle
    indices = list(range(len(self.filenames)))
    random.seed(self.random_seed)
    random.shuffle(indices)
    filenames = []
    classes = []
    hflips = []
    crops = []
    for idx in indices:
      filenames.append(self.filenames[idx])
      classes.append(self.classes[idx])
      hflips.append(self.hflips[idx])
      crops.append(self.crops[idx])
    self.filenames = filenames
    self.classes = classes
    self.hflips = hflips
    self.crops = crops
   
    # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
    self.batch_y = self.onehot(self.classes, self.num_class)

  def __len__(self):
    if self.class_aware_sampling == 2:
      count_value = max(self.sample_count) * self.num_class
      return math.ceil(count_value / float(self.batch_size))
    else:
      return math.ceil(len(self.filenames) / float(self.batch_size))
    
    
    
  # 补充队列中的数据
  def supplement_queue(self, supplement_class_idx):
    supplement_filenames = []
    for i in range(len(self.classes)):
      class_idx = self.classes[i]
      assert class_idx < len(self.filenames_cas)
      if class_idx == supplement_class_idx:
        filename = self.filenames[i]
        supplement_filenames.append(filename)

    random.shuffle(supplement_filenames)
    for filename in supplement_filenames:
      self.filenames_cas[supplement_class_idx].put_nowait(filename)


  # 生成batch的采样分布，在队列中采样，返回采样的文件名列表
  def construct_batch_class_aware_sampling(self, idx):
    # generate class_list
    class_list = []
    class_avg_samples = self.batch_size / self.num_class
    remain_samples = self.batch_size - self.num_class * class_avg_samples
    remain_samples_distribution = \
      random.sample(range(self.num_class), remain_samples)
    for class_idx in range(self.num_class):
      if class_idx in remain_samples_distribution:
        class_list.extend([class_idx] * (class_avg_samples + 1))
      else:
        class_list.extend([class_idx] * class_avg_samples)

    batch_size = len(class_list)
    random.shuffle(class_list)
    
    filenames = []
    for class_choose in class_list:
      filenames.append(self.filenames_cas[class_choose].get())

      global mutex_supplement_queue
      mutex_supplement_queue.acquire()
      if self.filenames_cas[class_choose].qsize() < self.min_queue_len:
        self.supplement_queue(class_choose)
      mutex_supplement_queue.release()

    return batch_size, filenames, class_list
  
  # https://github.com/facebookresearch/mixup-cifar10/blob/eaff31ab397a90fbc0a4aac71fb5311144b3608b/train.py#L119  
  def mixup_data(self, x, y, alpha=1.0):
    if alpha > 0:
      lam = np.random.beta(alpha, alpha)
    else:
      lam = 1
    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y, lam
    
  def __getitem__(self, idx):
    if self.class_aware_sampling:
      batch_size, filenames, temp_batch_y = \
        self.construct_batch_class_aware_sampling(idx)
      batch_x, batch_index = self.get_batch_x(filenames)
      # 该模式下暂不支持文件不存在，待修复
      assert batch_index == range(len(filenames)), 'file lost is not supported in cas mode'
      if self.optimizing_type == "classify":
        batch_y = self.onehot(temp_batch_y, self.num_class)
      else:
        batch_y = temp_batch_y
    else:
      start_idx = idx * self.batch_size
      end_idx = (idx + 1) * self.batch_size
      end_idx = min(end_idx, len(self.filenames))
      #batch_size = end_idx - start_idx
      batch_x, batch_index = self.get_batch_x(self.filenames[start_idx: end_idx], start_idx=start_idx)
      batch_index = [start_idx + index for index in batch_index]
      batch_y = self.batch_y[batch_index, :]
      if self.optimizing_type != "classify":
        batch_y = self.continues_code(batch_y)
    if self.mixup:
      batch_x, batch_y, _ = self.mixup_data(batch_x, batch_y, alpha=self.mixup_alpha)
    return batch_x, batch_y

  def continues_code(self, origin_code):
    class_list = []
    for onehot in origin_code:
      class_list.append(float(onehot.tolist().index(1)))
    return np.array(class_list)

  def onehot(self, origin_code, class_num):
    onehot_code = \
      np.zeros((len(origin_code), class_num), dtype=K.floatx())
    if self.label_smoothing == 0.:
      for i, label in enumerate(origin_code):
        onehot_code[i, label] = 1.
    else:
      if 0 <= self.label_smoothing <= 1:
        for i, label in enumerate(origin_code):
          onehot_code[i, label] = 1. * (1 - self.label_smoothing)
        onehot_code += self.label_smoothing / class_num
      else:
        raise Exception('Invalid label smoothing factor: ' + str(self.label_smoothing))
    return onehot_code

    
  def get_batch_x(self, filenames, start_idx=-1):
    process_lib = self.process_lib 
    batch_size = len(filenames)
    batch_x = np.zeros((batch_size,) + self.image_shape, dtype=K.floatx())
    batch_index = range(batch_size) # 每张图片最终对应的索引，有的图片可能会不存在，随机选择另外一张
    for i in range(batch_size):
      fname = filenames[i]
      # class_aware_sampling 暂时无法兼容 horizontal_flip 和 random_crop
      if start_idx == -1:
        hflip = False
        crop = (self.random_crop == 1)
      else:
        hflip = self.hflips[start_idx+i]
        crop = self.crops[start_idx+i] or (self.random_crop == 1)
      
      bbox_data = []
      if crop and self.random_crop_dict:
        if os.path.basename(fname) in self.random_crop_dict:
          bbox_data = self.random_crop_dict[os.path.basename(fname)]
        else:
          crop = False
      if self.list_mode == 0:
        fname = os.path.join(self.directory, fname)
      while not os.path.exists(fname):
        print('get_batch_x file not exist: %s' % fname)
        random_idx = random.randint(0, batch_size - 1)
        fname = filenames[random_idx]
        if self.list_mode == 0:
          fname = os.path.join(self.directory, fname)
        batch_index[i] = random_idx
      
      img = None
      crop_result = False
      # val_sequence存在，则将数据增强调整与val一致
      if self.val_sequence != None:
        x, crop_result = sm_load_img(path=fname,
                                       data_format=self.val_sequence.data_format,
                                       rotation_range=self.val_sequence.rotation_range,
                                       rotation90=self.val_sequence.rotation90,
                                       horizontal_flip=self.val_sequence.horizontal_flip,
                                       vertical_flip=self.val_sequence.vertical_flip,
                                       grayscale=False,
                                       target_size=self.val_sequence.target_size,
                                       interpolation=self.val_sequence.interpolation,
                                       letter_box=self.val_sequence.letter_box,
                                       levels=self.val_sequence.zoom_pyramid_levels,
                                       short_side=self.val_sequence.zoom_short_side,
                                       hflip=hflip,
                                       random_crop=crop,
                                       bbox_data=bbox_data,
                                       min_object_covered=self.val_sequence.min_object_covered,
                                       aspect_ratio_range=self.val_sequence.aspect_ratio_range,
                                       area_range=self.val_sequence.area_range,
                                       gaussian_blur=self.val_sequence.gaussian_blur,
                                       motion_blur=self.val_sequence.motion_blur,
                                       brightness=self.val_sequence.brightness,
                                       saturation=self.val_sequence.saturation,
                                       process_lib=process_lib)
      else:
        x, crop_result = sm_load_img(path=fname,
                                       data_format=self.data_format,
                                       rotation_range=self.rotation_range,
                                       rotation90=self.rotation90,
                                       horizontal_flip=self.horizontal_flip,
                                       vertical_flip=self.vertical_flip,
                                       grayscale=False,
                                       target_size=self.target_size,
                                       interpolation=self.interpolation,
                                       letter_box=self.letter_box,
                                       levels=self.zoom_pyramid_levels,
                                       short_side=self.zoom_short_side,
                                       hflip=hflip,
                                       random_crop=crop,
                                       bbox_data=bbox_data,
                                       min_object_covered=self.min_object_covered,
                                       aspect_ratio_range=self.aspect_ratio_range,
                                       area_range=self.area_range,
                                       gaussian_blur=self.gaussian_blur,
                                       motion_blur=self.motion_blur,
                                       brightness=self.brightness,
                                       saturation=self.saturation,
                                       process_lib=process_lib)
      
      if crop:
        self.random_crop_count += 1
        if crop_result:
          self.random_crop_success_count += 1.

      # save data augmentation
      if self.save_samples_num:
        if fname in self.save_data_augmentation_filenames:
          img_aug = array_to_img(x, data_format=self.data_format)
          save_name = '{prefix}_{fname}_{hash}.jpg'.format(prefix=self.save_prefix,
                                                       fname=os.path.basename(fname).split(".")[0],
                                                       hash=str(random.randint(0,10000)).zfill(5))
          label = fname.split("/")[-2]
          if not os.path.isdir(os.path.join(self.save_to_dir, label)):
            os.makedirs(os.path.join(self.save_to_dir, label))
          img_aug.save(os.path.join(self.save_to_dir, label, save_name))
      
      x = self.preprocessing_function(x)
      batch_x[i] = x

    return batch_x, batch_index

      
  def on_epoch_end(self):
    if self.shuffle and (not self.class_aware_sampling):
      self.shuffle_data()
    if self.random_crop:
      random_crop_rate = -1.0
      if self.random_crop_count != 0:
        random_crop_rate = self.random_crop_success_count/self.random_crop_count
      print("%s random crop imgs count: %d, success rate: %.4f" %
          (self.directory, self.random_crop_count, random_crop_rate))
      self.random_crop_count = 0
      self.random_crop_success_count = 0.
