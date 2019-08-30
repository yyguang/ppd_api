#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <teng.cao@foxmail.com> 2018-02-02 18:57:54
import os
import sys
import shutil
import datetime
import random
from PIL import Image as pil_image

# 将某个目录下所有图片随机采样一定数量到一个新的目录
# 第一个参数为图片目录
# 第二个参数为新的图片目录
# 第三个参数为要随机采样的数量
if __name__ == "__main__":
  img_dir = sys.argv[1]
  dst_dir = sys.argv[2]
  sample_num = int(sys.argv[3])
  suffix = ".jpg" # 图片类型不可知，统一存成jpg

  assert os.path.exists(img_dir)
  #count_fail_open = 0
  #count_succeed = 0
  all_files = []
  # os.walk() 返回包含隐藏文件
  for root, dirs, files in os.walk(img_dir):
    for file in files:
      file_name,file_ext = os.path.splitext(file)
      # 过滤.jpg文件
      if file_ext != '.jpg':
        continue
      else:
        fullpath = os.path.join(img_dir, file)
        #print('process file %s' % file)  
        #try:
        #  pil_image.open(open(fullpath, 'rb')).load()
        #except Exception as e:
        #  count_fail_open += 1
        #  print('error in open image %s: %s' % (fullpath, e))
        #  continue
        all_files.append(file)

  sample_num = min(sample_num, len(all_files))
  sampled = random.sample(all_files, sample_num)
  for f in sampled:
    src_fn = os.path.join(img_dir, f)
    dst_fn = os.path.join(dst_dir, f)
    assert not os.path.exists(dst_fn)
    shutil.copyfile(src_fn, dst_fn)
    #count_succeed += 1
  #print("  count_fail_open:", count_fail_open)
  #print("  count_succeed:", count_succeed)
