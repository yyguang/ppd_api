#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <teng.cao@foxmail.com> 2018-02-02 18:57:54
import os
import sys
import shutil
import datetime
import hashlib
from PIL import Image as pil_image
import getpass

# calc md5 for a file
def calc_md5(filepath):
  with open(filepath,'rb') as f:
    md5obj = hashlib.md5()
    md5obj.update(f.read())
    hash = md5obj.hexdigest()
    return hash

# 将某个目录下所有图片按md5进行重命名
# 第一个参数为图片目录
# 
if __name__ == "__main__":
  img_dir = sys.argv[1]
  suffix = ".jpg" # 图片类型不可知，统一存成jpg
  user = getpass.getuser()
  sudo = False
  if user == 'root' or user == 'caoteng':
    sudo = True
    print('sudo will be enabled for current user: %s' % user)

  assert os.path.exists(img_dir)
  count_md5_same = 0
  count_fail_open = 0
  count_succeed = 0
  # os.walk() 返回包含隐藏文件
  for root, dirs,files in os.walk(img_dir):
    for file in files:
      file_name,file_ext = os.path.splitext(file)
      # 过滤.jpg文件
      if file_ext != '.jpg':
        continue
      else:
        fullpath = os.path.join(img_dir, file)
        #print('process file %s' % file)  
        try:
          pil_image.open(open(fullpath, 'rb')).load()
        except Exception as e:
          count_fail_open += 1
          print('error in open image %s: %s' % (fullpath, e))
          continue

        md5 = calc_md5(fullpath)
        dst_fn = os.path.join(img_dir, md5+suffix)
        if os.path.exists(dst_fn) and dst_fn == fullpath:
          print('skip %s' % fullpath)
        elif os.path.exists(dst_fn):
          count_md5_same += 1
          if sudo:
            os.system('sudo chattr -a "' + fullpath + '"')
          os.remove(fullpath)
          print('remove %s' % fullpath)
        else:
          count_succeed += 1
          #print('%s -> %s' %(fullpath, dst_fn))
          if sudo:
            os.system('sudo chattr -a "' + fullpath + '"')
          os.rename(fullpath, dst_fn)
    print("count_md5_same: %d" % count_md5_same)
    print("count_fail_open: %d" % count_fail_open)
    print("count_succeed: %d" % count_succeed)
