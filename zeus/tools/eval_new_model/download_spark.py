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

urllist = sys.argv[1]
image_path = sys.argv[2] #urllist.split(".")[0]
md5_dict = {}
insert_dict = {}
md5_check = True # True，执行md5检查和插入
rename_md5 = True # 采用md5进行重命名，当执行数据修正时，需要保留原始文件名（文件名中保留了原始目录信息）

def download(url, label, md5_check, rename_md5):
  try:
    pic = requests.get(url[0], timeout = 100)
    fmd5 = hashlib.md5(pic.content).hexdigest()
    if (pic.status_code != 200):
      print('error download pic')
      pic.raise_for_status()
    elif (pic.headers['Content-Type'] not in ['image/jpeg','image_jpeg','image/png','image/x-ms-bmp','jpg','image/webp','image/jpg','application/octet-stream']):
      print('download pic header is not jpg',pic.headers['Content-Type'])
      raise TypeError
    elif (fmd5 in md5_dict):
      print('repeat' + '\t' + url[0] + "\t" + md5_dict[fmd5])
    else:
      md5_dict[fmd5] = url
      #label = image_path.split('/')[-1].strip()
      line = "insert into md5_lib values ('" + str(fmd5) + "'" + ",1,'" + "/data/project/xing/new_data_set/" + image_path + "/" + str(fmd5) + ".jpg','" + label +  "')"
      insert_dict[fmd5] = line
      if(len(insert_dict) % 500 == 0):
        print(len(insert_dict),time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())))
      if rename_md5:
        name = os.path.join(image_path, label, str(fmd5) + '_'+ url[1] + '_' +  str(url[2]) + '.jpg')
      else:
        name = os.path.join(image_path, label, os.path.basename(url[0]))
      fp = open(name, 'wb')
      fp.write(pic.content)
      fp.close()
      try:
        file = open(name, "rb")
        img = Image.open(file)
        file.close()
        #Image.open(name).load()
      except Exception as e:
        print('exception open image: %s, msg: %s, url %s' % (name, e, url[0]))
        os.remove(name)

  except Exception as e:
    print('exception download url: %s, msg: %s' % (url, e))

if os.path.exists(image_path):
  os.system("rm -r {}".format(image_path))
os.makedirs(image_path)

pool = ThreadPoolExecutor(max_workers=200)
future_list = []
urls = {}
with open(urllist, "r") as f:
  count=0
  for line in f:
    count += 1
    line = line.strip()
    parts = line.split()
    urls[parts[0],parts[1],parts[2]] = parts[-2]

for v in set(urls.values()):
  if not os.path.exists(image_path+"/"+v):
    os.makedirs(image_path+"/"+v)

for data in urls:
  future1 = pool.submit(download, data, urls[data], md5_check, rename_md5)
  future_list.append(future1)

wait(future_list)
#if md5_check:
#  for md5 in insert_dict:
#    cur.execute(insert_dict[md5])
#    conn.commit()
