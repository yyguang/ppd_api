#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author: chenyunkuo2018-10-30 20:58:08
# 对支持random crop的图片进行copy，可选择copy的类别，份数。
# Usage: python cropped_class_copy.py --classes exposed,normal --copy_times 1
# 说明: 对支持random crop的exposed、normal类别的图片在train的list中多copy一份(出现2次).

import os
import sys
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--crop_list', type=str, default='md5_random_crop.list',
    help='The img list file with bounding box info, which would be random cropped.'
  )
  parser.add_argument(
    '--dataset', type=str, default='train',
    help='The dataset list.'
  )
  parser.add_argument(
    '--classes', type=str, default='',
    help='The classes to be copied, default empty string mean that all classes would be copied. e.g. "exposed,porn".'
  )
  parser.add_argument(
    '--copy_times', type=int, default=1,
    help='"1" means that there will be double imgs with class copied.'
  )
  parser.add_argument(
    '--output', type=str, default='train.new',
    help='The output file.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  print(FLAGS)
  if len(unparsed) > 0:
    print("there are unknow args %s " % ','.join(unparsed))
    sys.exit(-1)
  
  crop_set = set()
  classes = set()
  if FLAGS.classes != '':
    classes = set(FLAGS.classes.split(','))

  for row in open(FLAGS.crop_list, 'r'):
    crop_set.add(row.strip().split()[0])
  print("crop imgs count: ", len(crop_set))
 
  row_count = 0
  add_count = 0 
  with open(FLAGS.output, 'w') as f:
    for row in open(FLAGS.dataset, 'r'):
      f.write(row)
      row_count += 1
      p, label = row.strip().split()
      if FLAGS.classes == '':
        classes.add(label)
      if label in classes:
        if os.path.basename(p) in crop_set:
          for i in range(FLAGS.copy_times):
            f.write(row)
            add_count += 1
  print("{} list row count : {}".format(FLAGS.dataset, row_count))
  print("new add row count : {}".format(add_count))
