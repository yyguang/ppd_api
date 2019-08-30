#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <teng.cao@foxmail.com> 2018-02-02 18:57:54
import os
import sys
import shutil
import datetime
import urllib
import hashlib
from PIL import Image as pil_image
import socket
socket.setdefaulttimeout(30) # 设置图片下载超时为30s

# calc md5 for a file
def calc_md5(filepath):
  with open(filepath,'rb') as f:
    md5obj = hashlib.md5()
    md5obj.update(f.read())
    hash = md5obj.hexdigest()
    return hash

# download an image given url and saved filename
def download_image(url, filename):
  try:
    urllib.urlretrieve(url, fn)
  except Exception as e:
    print('error in download image:', e)
    return False
  return True

# 从网页抓取得到csv后下载图像数据，并重命名为md5
# 第一个参数为csv文件所在的目录
# 第二个参数为要保存的目录，比如porn
if __name__ == "__main__":
  csv_dir = sys.argv[1]
  dst_dir = sys.argv[2]
  suffix = ".jpg" # 图片类型不可知，统一存成jpg

  if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

  count_fail_download = 0
  count_md5_same = 0
  count_fail_open = 0
  files = os.listdir(csv_dir)
  for file in files:
    fullpath = os.path.join(csv_dir, file)
    if not fullpath.endswith('.csv'):
      continue
    print('process file %s' % file)
    f = open(fullpath, 'r')
    f.readline()
    for line in f:
      parts = line.split(',')
      url = parts[3]
      assert url.startswith('http')
      fn = os.path.join(dst_dir, 'tmp')
      if not download_image(url, fn):
        count_fail_download += 1
        print('file download fail: %s' % url)
        continue # 下载不成功，跳到下一张
      
      try:
        pil_image.open(open(fn, 'rb')).load()
      except Exception as e:
        count_fail_open += 1
        print('error in open image %s: %s' % (url, e))
        continue
      
      md5 = calc_md5(fn)
      dst_fn = '%s/%s%s' % (dst_dir, md5, suffix)
      if os.path.exists(dst_fn):
        count_md5_same += 1
      else:
        os.rename(fn, dst_fn)
    f.close()
    if os.path.exists(fn)
      os.remove(fn)
  print("  count_fail_download:", count_fail_download)
  print("  count_md5_same:", count_md5_same)
  print("  count_fail_open:", count_fail_open)
