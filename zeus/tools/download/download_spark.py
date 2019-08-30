#!/usr/bin/env python
#-*- coding:utf8 -*-

#import MySQLdb
import json
import numpy as np
import requests
import sys
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import time
import os
from PIL import Image
import threading

urllist = os.path.join(sys.argv[1])
image_path = sys.argv[2]
md5_dict = {}
insert_dict = {}
md5_check = False # True，执行md5检查和插入
rename_md5 = True # 采用md5进行重命名，当执行数据修正时，需要保留原始文件名（文件名中保留了原始目录信息）
remove_duplicate = True # 删除重复的图片
mutex = threading.Lock()
headers = {
    'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36'}
def download(url, md5_check, rename_md5):
  try:
    pic = requests.get(url, timeout = 100, headers=headers)
    fmd5 = hashlib.md5(pic.content).hexdigest()
    fmd5_exist = False
    if mutex.acquire():
      if fmd5 not in md5_dict:
        md5_dict[fmd5] = 1
      else:
        md5_dict[fmd5] += 1
        fmd5_exist = True
      mutex.release()
    if (pic.status_code != 200):
      print('error download pic')
      pic.raise_for_status()
    elif (pic.headers['Content-Type'] not in ['image/jpeg','image_jpeg','image/png','image/webp','image/x-ms-bmp','jpg','image/jpg','application/octet-stream']):
      print('download pic header is not jpg')
      raise TypeError
    elif remove_duplicate and fmd5_exist:
      print('repeat' + '\t' + url + "\t")
    else:
      #md5_dict[fmd5] = url
      label = image_path.split('/')[-1].strip()
      line = "insert into md5_lib values ('" + str(fmd5) + "'" + ",1,'" + "/data/project/xing/new_data_set/" + image_path + "/" + str(fmd5) + ".jpg','" + label +  "')"
      insert_dict[fmd5] = line
      if(len(insert_dict) % 500 == 0):
        print(len(insert_dict),time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())))
      if rename_md5:
        if pic.headers['Content-Type']=='image/webp':
          name = os.path.join(image_path, str(fmd5) + os.path.splitext(url))
        else:
          name = os.path.join(image_path, str(fmd5) + '.jpg')
      else:
        name = os.path.join(image_path, os.path.basename(url))
      fp = open(name, 'wb')
      fp.write(pic.content)
      fp.close()
      try:
        Image.open(open(name, 'rb')).load()
      except Exception as e:
        print('exception open image: %s, msg: %s, url %s' % (name, e, url))
        os.remove(name)

  except Exception as e:
    print('exception download url: %s, msg: %s' % (url, e))

if not os.path.exists(image_path):
  os.makedirs(image_path)

pool = ThreadPoolExecutor(max_workers=200)
future_list = []
urls = []
with open(urllist, "r") as f:
  count=0
  for line in f:
    count += 1
    line = line.strip()[1:-1]
    parts = line.split(',')
    for part in parts:
      if part.startswith('http'):
        urls.append(part)

for data in urls:
  future1 = pool.submit(download, data, md5_check, rename_md5)
  future_list.append(future1)

wait(future_list)
f = open(image_path + '_md5_dict.txt', 'w')
for key,value in md5_dict.items():
  f.write(key + '\t' + str(value) + '\n')
f.close()
