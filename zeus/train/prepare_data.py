#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author Tony Tsao <cao-teng-teng.cao@foxmail.com> 2017-10-11 14:29:30

from __future__ import print_function
import sys
import tensorflow as tf
from tensorflow.python.platform import gfile
import re
import random
import os.path
import shutil
import hashlib
import string
import datetime
import math
from PIL import Image
import threading
import argparse
import cv2
import glob

# Seed for repeatability.
_RANDOM_SEED = 0

# input arguments
FLAGS = None

#check image size is less than min_size
def check_image(full_path, min_size):
  if min_size <= 0:
    return True
  try:
    im = Image.open(full_path)
    im.load()
    if (im.height < min_size or im.width < min_size):
      print('size img file is less then min_size, skip, img_file', full_path)
      return False
    else:
      return True
  except Exception as e:
    print('error img file   ',  str(e)) 
    return False

def load_sample_dict(sample_list_path, sample_dict=None):
  if not os.path.exists(sample_list_path):
    print('Error family list %s not exists' % (sample_list_path))
    exit(-1)
  if sample_dict is None:
    sample_dict = dict()
  with open(sample_list_path, 'r') as sample_list_f:
    for line in sample_list_f.readlines():
      img_md5 = line.split(',')[0]
      img_is_sample = int(line.split(',')[1])
      if img_md5 in sample_dict:
        print('Error same md5: %s' % (img_md5))
      else:
        sample_dict[img_md5] = img_is_sample

  return sample_dict

# calc md5 for a file
def calc_md5(filepath):
  with open(filepath,'rb') as f:
    md5obj = hashlib.md5()
    md5obj.update(f.read())
    hash = md5obj.hexdigest()
    return hash

# generate a random string with specified len n
def generate_random_string(n):
  seed = string.ascii_letters + string.digits
  random_chars = []
  for i in range(n):
    random_chars.append(random.choice(seed))
  random_str = ''.join(random_chars)
  return random_str

# resize an image with its short side set to short_side
def resize_image(filename, short_side):
  img = cv2.imread(filename)  
  height, width = img.shape[:2]
  if height < width:
    height2 = short_side
    width2 = float(height2) / height * width
  else:
    width2 = short_side
    height2 = float(width2) / width * height
  size = (int(round(width2)), int(round(height2)))
  resized = cv2.resize(img, size)
  cv2.imwrite(filename, resized)


# this function random copy from src_files to dst_dir, with a total
# of total_number files, filename is random generated and unique
def oversampling_files(src_files, dst_dir, total_number):
  num_src = len(src_files)
  len_of_filename = 64
  random.seed(_RANDOM_SEED)
  for i in range(total_number):
    select = random.randint(0, num_src-1)
    src_file = src_files[select]
    while True:
      basename = generate_random_string(len_of_filename)
      dst_name = dst_dir + '/' + basename + os.path.splitext(src_file)[1]
      if not os.path.exists(dst_name): 
        break
    shutil.copyfile(src_file, dst_name)

# md5_dict:  md5 -> filename
# md5_label: md5 -> label
# basename_dict:  basename -> filename
# basename_label: basename -> label
# all files copied are renamed with md5
# this function checks for file unique, wrong labeling for md5 and basename
def copy_files(src_files, dst_dir, md5_dict, basename_dict, md5_label,
      basename_label, label, rotation_variations, hflip_variations, md5_check=True,
      sample_dict=None):               
  if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
  same_basename_files = 0
  same_md5_files = 0
  clean_files = 0
  for f in src_files:
    rst = check_image(f,FLAGS.min_size)
    if not rst:
      continue
    ext = os.path.splitext(f)[1].lower()
    if ext != '.jpg' and ext != '.jpeg' and ext != '.png':
      print('find a not-image file: ' + f)
      #sys.exit(0)
    md5 = calc_md5(f)

    # hash clean
    if sample_dict is not None:
      if md5 not in sample_dict:
        print('Error md5 %s not in sample_dict' % (md5))
      else:
        if not sample_dict[md5]:
          clean_files += 1
          continue
      
    (_, tempfilename) = os.path.split(f);  
    (shortname, _) = os.path.splitext(tempfilename);
    if md5 != shortname:
      print('filename different with md5: ' + f + ' and ' + md5)
    if md5_check:
      if md5 in md5_dict:
        same_md5_files += 1
        print('same md5 for file: ' + f + ', and ' + md5_dict[md5])
        #sys.exit(0)
      if md5 in md5_label:
        if md5_label[md5] != label:
          print('same md5 different label for file ' + f + ' and ' + md5_dict[md5])
          #sys.exit(0)
      basename = os.path.basename(f)
      if basename in basename_dict:
        same_basename_files += 1
      if basename in basename_label:
        if basename_label[basename] != label:
          print('same basename different label for file ' + f + ' and ' +
            basename_dict[basename])
          #sys.exit(0)
      md5_dict[md5] = f
      md5_label[md5] = label
      basename_dict[basename] = f
      basename_label[basename] = label
      #shutil.copy(f, dst_dir)
    dst_name = dst_dir + '/' + md5 + os.path.splitext(f)[1]
    if not os.path.exists(f):
      print('\tsrc file not exist: ' + f)
      sys.exit(0)
    if os.path.exists(dst_name):
      print('\tdst file exist: ' + dst_name)
  
    #try:
    #  img_file = open(f, "rb")
    #  img = Image.open(img_file)
    #  new = img.resize((1, 1), Image.BILINEAR)
    #  img_file.close()
    #except Exception as e:
    #  print('exception open image: %s' % f)
    shutil.copyfile(f, dst_name)
    if FLAGS.resize_length:
      resize_image(dst_name, FLAGS.resize_length)
    if hflip_variations:
      hflip_image(dst_name)

  # hash clean
  print('\tclean: %d imgs' % (clean_files))
  
  if same_md5_files != 0:
    print('\tsame_md5_files: ' + str(same_md5_files))
  if same_basename_files != 0:
    print('\tsame_basename_files: ' + str(same_basename_files))
  
  # process rotation variations
  if rotation_variations:
    print('\tstart generating rotation variations...')
    generate_rotation_variations(dst_dir)

# thread for worker
class myThread(threading.Thread):
  def __init__(self, image_list):
    threading.Thread.__init__(self)
    self.image_list = image_list
  def run(self):
    do_rotate(self.image_list)

# thread function
def do_rotate(image_list):
  for fullpath in image_list:
    prefix = os.path.splitext(fullpath)[0]
    ext = os.path.splitext(fullpath)[1].lower()
    img = Image.open(fullpath)
    if len(img.getbands()) != 3:
      img = img.convert('RGB')
    img_rotate90 = img.rotate(90)
    img_rotate90.save(prefix + '_rotate90' + ext)
    img_rotate180 = img.rotate(180)
    img_rotate180.save(prefix + '_rotate180' + ext)
    img_rotate270 = img.rotate(270)
    img_rotate270.save(prefix + '_rotate270' + ext)

# do hflip and save
def hflip_image(filename):
  prefix = os.path.splitext(filename)[0]
  ext = os.path.splitext(filename)[1].lower()
  img = Image.open(filename)
  if len(img.getbands()) != 3:
    img = img.convert('RGB')
  out = img.transpose(Image.FLIP_LEFT_RIGHT)
  out.save(prefix+ '_hflip' + ext)

# generate rotation variations for all images in a specified dir
def generate_rotation_variations(path, thread_num = 16):
  image_list = []
  for dirpath, dirnames, filenames in os.walk(path):
    for file in filenames:
      fullpath = os.path.join(dirpath, file)
      ext = os.path.splitext(fullpath)[1].lower()
      if ext == '.jpg' or ext == '.jpeg':
        image_list.append(fullpath)
  
  files_per_thread = math.ceil(len(image_list) / float(thread_num))
  files_per_thread = int(files_per_thread)
  threads = []
   
  for i in range(thread_num):
    start_idx = files_per_thread * i
    end_idx = files_per_thread * (i+1)
    end_idx = min(end_idx, len(image_list))
    thread1 = myThread(image_list[start_idx:end_idx])
    thread1.start()
    threads.append(thread1)
  for t in threads:
    t.join()

# 
def process_and_copy_files(key, training_images, validation_images,
    testing_images, md5_dict, basename_dict, md5_label, basename_label,
    oversampling_number_training = 0, oversampling_number_validation = 0,
    rotation_variations = 0, hflip_variations = 0,
    sample_dict=None):
  print("label %s:" % key)
  train_dir = 'tmp/train/' + key
  validation_dir = 'tmp/validation/' + key
  test_dir = 'tmp/test/' + key
  print('\ttraining images total %d, start process and copy...' %
      (len(training_images)))
  copy_files(training_images, train_dir, md5_dict, basename_dict, md5_label,
    basename_label, key, rotation_variations, hflip_variations,
    sample_dict=sample_dict)
  print('\tvalidation images total %d, start process and copy...' %
      (len(validation_images)))
  copy_files(validation_images, validation_dir, md5_dict, basename_dict,
    md5_label, basename_label, key, rotation_variations, hflip_variations,
    sample_dict=sample_dict)             
  print('\ttesting images total %d, start process and copy...' % (len(testing_images)))
  copy_files(testing_images, test_dir, md5_dict, basename_dict, md5_label,
    basename_label, key, 0, 0,
    sample_dict=sample_dict)  # generate variants only for training images             
  if oversampling_number_training != 0:
    print('\toversampling %d trainging images...' % oversampling_number_training)
    oversampling_files(training_images, train_dir, oversampling_number_training)
  if oversampling_number_validation != 0:
    print('\toversampling %d validation images...' % oversampling_number_validation)
    oversampling_files(validation_images, validation_dir, oversampling_number_validation)
  
# usage
def usage():
  print('Usage #1: create data for keras from image list files, e.g. \
shumei-porn-1709-training.list. In this config, the validation set is \
sampled from all of the files in the list.')
  print('%s training.list test.list validation_percentage oversampling rotation_variations' % (sys.argv[0]))
  print('\tline format of the list file is: fullpath label')
  print('Usage #2: create data for keras from dataset files, e.g. \
shumei-porn-1709-training.txt. In this config, the validation set is \
sampled from each database folder, so the distribution of the \
diversity should be more similar with the training data')
  print('%s training.txt test.txt validation_percentage oversampling rotation_variations' % (sys.argv[0]))
  print('\tline format of the dataset file is: fullpath_of_a_databse')
  print('the default setting of validation_percentage is 10')
  print('the default setting of oversampling is 0, which means False')
  print('the default setting of rotation_variations is 0, whhich means no \
rotations variations(90 180 and 270 degrees) will be added')
  print('Usage #3: automatic create training dataset file for base dataset plus \
all diff files, and then prepare the data like usage #2')
  print('%s dataset_dir training-base.txt test.txt validation_percentage oversampling rotation_variations')
  print('the diff files should be with the pattern of: diff_YYYYMMDD')

# create from list file
def create_from_list(image_list_training, image_list_test, validation_percentage):
  if not gfile.Exists(image_list_training):
    print("Image list for training '" + image_list_training + "' not found.")
    sys.exit(0)
  if not gfile.Exists(image_list_test):
    print("Image list for test '" + image_list_test + "' not found.")
    sys.exit(0)
  
  label_images_dict = {}
  test_images_dict = {}  
  with open(image_list_training, "r") as f:
    for line in f:
      vec = line.strip().split()
      if len(vec) != 2:
        print("line format error in file '" + image_list_training + "': " + line)
        sys.exit(0)
      label = re.sub(r'[^a-z0-9]+', ' ', vec[1])
      if not label in label_images_dict:
        label_images_dict[label] = []
      label_images_dict[label].append(vec[0])
  with open(image_list_test, "r") as f:
    for line in f:
      vec = line.strip().split()
      if len(vec) != 2:
        print("line format error in file '" + image_list_test + "': " + line)
        sys.exit(0)
      label = re.sub(r'[^a-z0-9]+', ' ', vec[1])
      if not label in test_images_dict:
        test_images_dict[label] = []
      test_images_dict[label].append(vec[0])
  
  md5_dict = {}
  md5_label = {}
  basename_dict = {}
  basename_label = {}
  for key, value in label_images_dict.items():
    # rand the training files 
    #random.seed(_RANDOM_SEED)
    #random.shuffle(value)
    value.sort(key=lambda path:os.path.basename(path))
    val_num = len(value) * validation_percentage / 100.0
    val_num = int(round(val_num))
    validation_images = value[:val_num] # 随机后的前百分之x的数据作为验证集
    training_images = value[val_num:]
    testing_images = test_images_dict[key];
    process_and_copy_files(key, training_images, validation_images, testing_images, md5_dict, basename_dict, md5_label, basename_label)

def get_images_from_image_dir(image_dir, is_training, validation_percentage, train_images_dict, validation_images_dict, test_images_dict, last_val_set):
  if not gfile.Exists(image_dir):
    print("Image dir '" + image_dir + "' not found.")
    return None
  # Ensure all bad case flow into train dataset
  if 'bad_case' in image_dir or 'badcase' in image_dir:
    validation_percentage = 0.
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    #print("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))
      #file_list.sort()
    if not file_list:
      print('Warning: No files found in dir: ' + sub_dir)
      continue
    #label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    label_name = dir_name.lower()
    #random.seed(_RANDOM_SEED)
    #random.shuffle(file_list)
    file_list.sort(key=lambda path:os.path.basename(path))
    if is_training:
      val_num = len(file_list) * validation_percentage / 100.0
      val_num = int(round(val_num))
      validation_images = []
      training_images = []
      if len(last_val_set) == 0:
        validation_images = file_list[:val_num]
        training_images = file_list[val_num:]
      else:
        for img in file_list:
          basename = os.path.basename(img)
          if len(validation_images) < val_num and basename not in last_val_set:
             validation_images.append(img)
          else:
             training_images.append(img)
      if not label_name in train_images_dict:
        train_images_dict[label_name] = []
      train_images_dict[label_name].extend(training_images)
      if not label_name in validation_images_dict:
        validation_images_dict[label_name] = []
      validation_images_dict[label_name].extend(validation_images)
    else:
      testing_images = file_list
      if not label_name in test_images_dict:
        test_images_dict[label_name] = []
      test_images_dict[label_name].extend(testing_images)
    
# merge data from cat_from into cat_to. e.g. merge data from complete_normal into normal
def merge_category(images_dict, cat_from, cat_to, infor):
  if cat_from in images_dict:
    if not cat_to in images_dict:
      print('class %s not exists in image dir, and it has been created now.' % (cat_to))
      images_dict[cat_to] = []
    images_from = images_dict[cat_from]
    print('%s: merge category %s with count %d into %s' % (infor, cat_from,
          len(images_from), cat_to))
    images_dict[cat_to].extend(images_from)


def collect_category_md5(md5_list, images_dict, category_name,
                         infor):
  if category_name in images_dict:
    images_collected_category = images_dict[category_name]
    print('%s: collect category %s with count %d'
          % (infor, category_name, len(images_collected_category)))
    for image in images_collected_category:
      md5_list.append(os.path.basename(image))
    # 去重，或者考虑改用 set
    # md5_list = list(set(md5_list))


def save_category_md5(md5_list, dst_name, category_name):
  print('save image\'s md5 of category %s to %s'
        % (category_name, dst_name))
  with open(dst_name, 'w') as md5_list_file:
    for image_md5 in md5_list:
      md5_list_file.write(str(image_md5)+'\n')
    

# create from dataset file
def create_from_dataset(dataset_training, dataset_test, validation_percentage,
      oversampling, rotation_variations, hflip_variations, is_hash_clean):
  if not gfile.Exists(dataset_training):
    print("Trainging dataset file '" + dataset_training + "' not found.")
    return None
  if dataset_test and not gfile.Exists(dataset_test):
    print("Test dataset file '" + dataset_test + "' not found.")
    return None
  if not os.path.exists('tmp'):
    os.makedirs('tmp')
  dst_name = 'tmp/training_' + os.path.basename(dataset_training)
  shutil.copyfile(dataset_training, dst_name)
  if dataset_test:
    dst_name = 'tmp/test_' + os.path.basename(dataset_test)
    shutil.copyfile(dataset_test, dst_name)
  last_val_set = set()
  if FLAGS.last_validation:
    if os.path.isdir(FLAGS.last_validation):
      for img in glob.glob(os.path.join(FLAGS.last_validation, "*", "*.jpg")):
        last_val_set.add(os.path.basename(img))
    elif os.path.exists(FLAGS.last_validation):
      for row in open(FLAGS.last_validation, 'r'):
        last_val_set.add(os.path.basename(row.strip().split()[0]))

  train_images_dict = {} 
  validation_images_dict = {}
  test_images_dict = {}
  with open(dataset_training, "r") as f:
    for line in f:
      image_dir = line.strip()
      get_images_from_image_dir(image_dir, True, validation_percentage, train_images_dict, validation_images_dict, test_images_dict, last_val_set)
  if dataset_test:
    with open(dataset_test, "r") as f:
      for line in f:
        image_dir = line.strip()
        get_images_from_image_dir(image_dir, False, validation_percentage, train_images_dict, validation_images_dict, test_images_dict, last_val_set)

  # init sample dict
  if is_hash_clean:
    train_list_name = os.path.basename(dataset_training).split('.')[0]
    test_list_name = os.path.basename(dataset_test).split('.')[0]
    train_sample_list_file_name = train_list_name + '.sample.list'
    test_sample_list_file_name = test_list_name + '.sample.list'
    train_sample_list_path = os.path.join('../tools/hash_clean/' + train_sample_list_file_name)
    test_sample_list_path = os.path.join('../tools/hash_clean/' + test_sample_list_file_name)

    sample_dict  = dict()
    load_sample_dict(train_sample_list_path, sample_dict)
    load_sample_dict(test_sample_list_path, sample_dict)
  else:
    sample_dict = {}
      
  ignore_list = FLAGS.ignore_labels.split(',')
  print('ignore_list: %s' % ignore_list)
  special = FLAGS.special
  classes_map = FLAGS.classes_map
  # 读取classes_map文件中的类别融合关系
  if classes_map:
    assert os.path.exists(classes_map)
    classes_map_dict = {}
    for row in open(classes_map, 'r'):
      assert len(row.strip().split()) == 2
      ori_label, new_label = row.strip().split()
      classes_map_dict[ori_label] = new_label
    ignore_classes = set(classes_map_dict.keys()) - set(classes_map_dict.values())
    print(classes_map_dict)
    print([i for i in ignore_classes])

  # 对于porn分类的特殊处理为：将complete_normal加入normal中，并删除complete_normal类型
  if special == 'porn':
    # 收集指定类别文件的 md5
    complete_normal_md5 = []
    dst_name = 'tmp/md5_random_crop.list'    
    collect_category_md5(complete_normal_md5, train_images_dict,
                         'complete_normal', 'train')
    collect_category_md5(complete_normal_md5, validation_images_dict,
                         'complete_normal', 'validation')
    collect_category_md5(complete_normal_md5, test_images_dict,
                         'complete_normal', 'test')
    save_category_md5(complete_normal_md5, dst_name, 'complete_normal')
    if classes_map:
      for cls in ignore_classes:
        merge_category(train_images_dict, cls, classes_map_dict[cls], 'train')
        merge_category(validation_images_dict, cls, classes_map_dict[cls], 'validation')
        merge_category(test_images_dict, cls, classes_map_dict[cls], 'test')
        ignore_list.append(cls)
  elif special == 'behavior':
    if classes_map:
      for cls in ignore_classes:
        merge_category(train_images_dict, cls, classes_map_dict[cls], 'train')
        merge_category(validation_images_dict, cls, classes_map_dict[cls], 'validation')
        merge_category(test_images_dict, cls, classes_map_dict[cls], 'test')
        ignore_list.append(cls)
  elif special == 'violence':
    merge_category(train_images_dict, 'kongbuzhuyibiaozhi', 'kongbuzuzhi', 'train')
    merge_category(validation_images_dict, 'kongbuzhuyibiaozhi', 'kongbuzuzhi', 'validation')
    merge_category(test_images_dict, 'kongbuzhuyibiaozhi', 'kongbuzuzhi', 'test')
    ignore_list.append('kongbuzhuyibiaozhi')
 
  # generate lists file for training  validation and test
  if FLAGS.list_mode:
    imgs_label = {}
    dst_fn ='tmp/train'
    f = open(dst_fn, 'w')
    for key, value in train_images_dict.items():
      if key not in ignore_list:
        for img in value:
          basename = os.path.basename(img)
          md5 = basename.split(".")[0]
          if md5 in sample_dict and not sample_dict[md5]:
            continue
          if basename not in imgs_label:
            imgs_label[basename] = key
          elif imgs_label[basename] != key:
            print("{} with diff label: {} {}".format(basename, key, imgs_label[basename]))
            #sys.exit(0)
          if check_image(img, FLAGS.min_size):
            f.write('%s\t%s\n' % (img, key))
          else:
            continue
    f.close()
    print('list file for training saved: %s' % (dst_fn))
    
    dst_fn = 'tmp/validation'
    f = open(dst_fn, 'w')
    for key, value in validation_images_dict.items():
      if key not in ignore_list:
        for img in value:
          basename = os.path.basename(img)
          md5 = basename.split(".")[0]
          if md5 in sample_dict and not sample_dict[md5]:
            continue
          if basename not in imgs_label:
            imgs_label[basename] = key
          elif imgs_label[basename] != key:
            print("{} with diff label: {} {}".format(basename, key, imgs_label[basename]))
            #sys.exit(0)
          if check_image(img, FLAGS.min_size):
            f.write('%s\t%s\n' % (img, key))
          else:
            continue
    f.close()
    print('list file for validation saved: %s' % (dst_fn))

    dst_fn = 'tmp/test'
    f = open(dst_fn, 'w')
    for key, value in test_images_dict.items():
      if key not in ignore_list:
        for img in value:
          basename = os.path.basename(img)
          md5 = basename.split(".")[0]
          if md5 in sample_dict and not sample_dict[md5]:
            continue
          if basename not in imgs_label:
            imgs_label[basename] = key
          elif imgs_label[basename] != key:
            print("{} with diff label: {} {}".format(basename, key, imgs_label[basename]))
            #sys.exit(0)
          if check_image(img, FLAGS.min_size):
            f.write('%s\t%s\n' % (img, key))
          else:
            continue
    f.close()
    print('list file for test saved: %s' % (dst_fn))
 
  md5_dict = {}
  md5_label = {}
  basename_dict = {}
  basename_label = {}
  training_images_max = 0
  validation_images_max = 0
  if FLAGS.list_mode:
    return 
  for key, value in train_images_dict.items():
    if key in ignore_list:
      continue
    training_images_max = max(training_images_max, len(value))
    validation_images_max = max(validation_images_max,
        len(validation_images_dict[key]))
  for key, value in train_images_dict.items():
    #if (not key in validation_images_dict) or (not
    #   key in test_images_dict):
    #  continue
    if key in ignore_list:
      continue
    # rand the training files 
    random.seed(_RANDOM_SEED)
    random.shuffle(value)
    training_images = value
    validation_images = validation_images_dict[key]
    if key in test_images_dict:
      testing_images = test_images_dict[key]
    else:
      testing_images = []
    if oversampling:
      oversampling_training = training_images_max - len(training_images)
      oversampling_validation = validation_images_max - len(validation_images)
    else:
      oversampling_training = 0
      oversampling_validation = 0
    process_and_copy_files(key, training_images, validation_images,
        testing_images, md5_dict, basename_dict, md5_label, basename_label,
        oversampling_training, oversampling_validation, rotation_variations,
        hflip_variations, sample_dict=sample_dict)


# main process
# the created dataset is in ./tmp dir
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--train_dataset', type=str,
    default='/data/project/xing/porn_data_set/daily-180827.txt',
    help='Path to the train dataset define txt file, each line specify  a root dir for a dataset.'
  )
  parser.add_argument(
    '--test_dataset', type=str,
    default='',
    help='Path to the test dataset define txt file, same format as train_dataset. And if value is empty, it means that using val as test.'
  )
  parser.add_argument(
    '--validation_percentage', type=int, default=10,
    help='Percentage of training data for validation.'
  )
  parser.add_argument(
    '--oversampling', type=int, default=0,
    help='Whether oversampling train and test images.'
  )
  parser.add_argument(
    '--rotation_variations', type=int, default=0,
    help='Whether generating rotation variations of 90, 180 and 270 degrees for train and test images.'
  )
  parser.add_argument(
    '--hflip_variations', type=int, default=0,
    help='Whether generating hflip variations for train and test images.'
  )
  parser.add_argument(
    '--ignore_labels', type=str,
    default='porn_not_sure,not_sure,vulgar,smoke_drink,buqueding', 
    help='Labels to ignore, seperate with comma.'
  )
  parser.add_argument(
    '--special', type=str, default='',
    help='Special process for different types of tasks, e.g. porn for porn recognition, violence for violence recognition.'
  )
  parser.add_argument(
    '--resize_length', type=int, default=0,
    help='Resize the short side of the image to a given length, retain ascpect.'
  )
  parser.add_argument(
    '--is_hash_clean', type=int, default=0,
    help='whether toggle on hash clean.'
  )
  parser.add_argument(
    '--min_size', type=int, default=10,
    help='The min size of pics in the data_set, and it does not work if min_size=0.'
  )
  parser.add_argument(
    '--list_mode', type=int, default=1,
    help='train.list,test.list,validation.list for train.默认值为不开启list_mode,如果开启的话把值设定为1'
  )
  parser.add_argument(
    '--classes_map', type=str, default="",
    help='输入类别映射的list文件，按照该文件进行类别融合及ignore被融合的类别，e.g. porn_3classes_map.list, porn_4classes_map.list 分别代表色情3、4分类的class map文件.'
  )
  parser.add_argument(
    '--last_validation', type=str, default="",
    help='输入一份已生成好的验证集，确保新生成的验证集与该集完全不同.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  print(FLAGS)
  if len(unparsed) > 0:
    print("there are unknow args %s " % ','.join(unparsed))
    sys.exit(-1)
  argc = len(sys.argv)

  # generate data from list or dataset txt file
  file_training = FLAGS.train_dataset
  file_test = FLAGS.test_dataset
  validation_percentage = FLAGS.validation_percentage
  oversampling = FLAGS.oversampling
  rotation_variations = FLAGS.rotation_variations
  hflip_variations = FLAGS.hflip_variations
  is_hash_clean = FLAGS.is_hash_clean
  if file_training.endswith('.list'):
    create_from_list(file_training, file_test, validation_percentage)
  else:
    create_from_dataset(file_training, file_test, validation_percentage,
      oversampling, rotation_variations, hflip_variations, is_hash_clean)
  if not file_test:
    print("Using validation as test.")
    shutil.copy("tmp/validation", "tmp/test")
  if FLAGS.special == 'porn' and os.path.exists("/data/project/xing/porn_data_set/porn_random_crop.list"):
    os.system("mv tmp/md5_random_crop.list tmp/complete_normal_random_crop.list")
    os.system("cat tmp/complete_normal_random_crop.list /data/project/xing/porn_data_set/porn_random_crop.list > tmp/md5_random_crop.list")
