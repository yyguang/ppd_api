#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <cao-teng-teng.cao@foxmail.com> 2018-03-22 15:42:30
# 输入数据集定义文件，自动进行去重，检查是否有损坏或不支持的图片

from __future__ import print_function
import sys
import re
import random
import os.path
import shutil
import hashlib
import string
import time
import math
from PIL import Image
import threading
import argparse
import glob

# calc md5 for a file
def calc_md5(filepath):
  with open(filepath,'rb') as f:
    md5obj = hashlib.md5()
    md5obj.update(f.read())
    hash = md5obj.hexdigest()
    return hash

def get_images_from_image_dir(image_dir, images_dict, save_remove_dir, log, with_remove=0,
    check_support=0):
  if not os.path.exists(image_dir):
    print("Image dir '" + image_dir + "' not found.")
    log.write("Image dir '" + image_dir + "' not found.\n")
    return None
  sub_dirs = os.listdir(image_dir) #[x[0] for x in gfile.Walk(image_dir)]
  for sub_dir in sub_dirs:
    fullpath = os.path.join(image_dir, sub_dir)
    if os.path.isfile(fullpath):
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG']
    file_list = []
    for extension in extensions:
      file_glob = os.path.join(fullpath, '*.' + extension)
      file_list.extend(glob.glob(file_glob))
      file_list.sort()
    label_name = sub_dir.lower()
    for f in file_list:
      if check_support:
        try:
          Image.open(open(f, 'rb')).load()
        except:
          print('exception open image: %s' % f)
      basename = os.path.basename(f)
      if basename in images_dict:
        value = images_dict[basename]
        if label_name == value[1]:
          if with_remove:
            os.system("mv {} {}".format(f, save_remove_dir))
          print('file %s same with %s, label: %s' % (f, value[0], label_name))
          log.write('file %s same with %s, label: %s\n' % (f, value[0], label_name))
        else:
          print('file %s label %s <-> file %s label %s' % (f, label_name,
                value[0], value[1]))
          log.write('file %s label %s <-> file %s label %s\n' % (f, label_name,
                value[0], value[1]))
      else:  
        images_dict[basename] = (f, label_name)
    
# main process
# 第一个参数为数据集定义文件，比如daily-180324.txt，里面每行对应一个数据库根目录
# 第二个参数为是否移除相同文件名的文件，一般先设置为0跑一遍，进行观察，确认没问题后再设置为1再跑一遍，移除相同文件名的文件
# 删除重复文件时务必按照以下准则进行：当两个文件label相同时，删除旧的；当两个文件label有冲突时，确认后删除错误的
# 第三个参数为是否要检查文件是否可支持，0表示不进行检查
if __name__ == '__main__':
  dataset = sys.argv[1]
  with_remove = int(sys.argv[2])
  check_support = int(sys.argv[3])
  if not os.path.exists(dataset):
    print("Dataset file '" + dataset + "' not found.")
    sys.exit(-1)
  timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
  save_remove_dir = '/data/project/xing/porn_data_set/check_data_remove'
  if not os.path.isdir(save_remove_dir):
    os.makedirs(save_remove_dir)
  images_dict = {}
  dir_set = set()
  with open(os.path.join("/data/project/xing/porn_data_set/", "check_data_{}.log".format(timestamp)), 'w') as log:
    with open(dataset, "r") as f:
      for line in f:
        image_dir = line.strip()
        if image_dir in dir_set:
          print("repeated: " + image_dir)
          continue
        dir_set.add(image_dir)
        print('\nprocess dir %s' % image_dir)
        get_images_from_image_dir(image_dir, images_dict, save_remove_dir, log, with_remove,
            check_support)
