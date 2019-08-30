#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <teng.cao@foxmail.com> 2018-01-30 17:56:34
import os
import sys
import glob
import shutil
import datetime

# 从dataset列表文件中读取原始图片路径
def read_dataset_list_file(dataset_list_file):
  img_path = {}
  for row in open(dataset_list_file, "r"):
    for img in glob.glob(os.path.join(row.strip(), "*", "*.jpg")):
      md5_jpg = img.split("/")[-1]
      img_path[md5_jpg] = img
  print('dataset img count: {}'.format(len(img_path)))
  return img_path

# 从dataset root目录读取img
def read_dataset(dataset_list_file):
  img_path = {}
  for img in glob.glob(os.path.join(dataset_list_file.strip(), "*/*", "*.jpg")):
    md5_jpg = img.split("/")[-1]
    img_path[md5_jpg] = img
  print('dataset img count: {}'.format(len(img_path)))
  return img_path

# 修正误分类数据到本地数据库
# 第一个参数为下载回来的重标注数据的根目录
# 第二个参数为原始数据库存放的根目录，比如对于暴恐而言，该参数为/data/project/xing/violence_data_set
# 第二个参数也可以是数据集定义文件，比如/data/project/xing/new_data_set/shumei-porn-1801-test.txt
# 第三个参数为是否确认执行文件操作，可选值为0或1，1表示执行文件操作
# 调用示例：python misclassify_correct.py /home/caoteng/data/porn-data-correct/misclassify_20180130_9847 /data/project/xing/violence_data_set
# 如果图片来源于gen*目录，不进行修正
if __name__ == "__main__":
  assert len(sys.argv) == 4
  data_dir = sys.argv[1]
  dataset_list_file = sys.argv[2]
  exe_confirm = int(sys.argv[3])
  if not os.path.exists(data_dir):
    print('data dir not exists: %s' % data_dir)
    sys.exit(-1)
  img_path = {}
  if os.path.isdir(dataset_list_file):
    img_path = read_dataset(dataset_list_file)
  else:
    img_path = read_dataset_list_file(dataset_list_file)
  
  correct_total = set()
  error_count = 0
  label_same = 0
  remove_duplicate = 0
  subdirs = os.listdir(data_dir)
  for subdir in subdirs:
    fulldir = os.path.join(data_dir, subdir)
    if not os.path.isdir(fulldir):
      continue
    print('process subdir %s' % subdir)
    new_cat = subdir
    files = os.listdir(fulldir)
    for f in files:
      fullpath = os.path.join(fulldir, f)
      if not os.path.isfile(fullpath):
        error_count += 1
        continue
      basename = f
      if f not in img_path: # 原目录中没有的图片
        print("{} not in dataset_list".format(f))
        error_count += 1
        continue
      if f in correct_total: # 重复修正的文件，一般不会出现
        print("{} repeated.".format(f))
        error_count += 1
        continue
      database, original_cat = img_path[f].split("/")[-3: -1]
      if database.startswith('gen'): # 生成图片不用修正
        continue
      if original_cat == new_cat: # label一致，不需要修正
        label_same += 1
        continue
      else:
        src_fn = img_path[basename]
        dst_dir = os.path.join("/".join(src_fn.split("/")[:-2]), new_cat)
        dst_fn = os.path.join(dst_dir, basename)
        #print('correct %s to %s' % (src_fn, dst_fn))
        correct_total.add(basename)
        assert os.path.exists(src_fn)
        if os.path.exists(dst_fn):
          print("remove duplicate {}".format(src_fn))
          remove_duplicate += 1
          if exe_confirm:
            os.system('sudo chattr -a "' + src_fn + '"') 
            os.remove(src_fn)
          continue
        if exe_confirm:
          if not os.path.exists(dst_dir):
            #print('makedirs %s' % dst_dir)
            os.makedirs(dst_dir)
          os.system('sudo chattr -a "' + src_fn + '"') 
          shutil.move(src_fn, dst_fn)
          os.system('sudo chattr +a "' + dst_fn + '"') 
          assert not os.path.exists(src_fn)
          assert os.path.exists(dst_fn)
  print('error count %d files' % error_count)
  print('label same %d files' % label_same)
  print('remove_duplicate %d files' % remove_duplicate)
  print('correct total %d files' % len(correct_total))
