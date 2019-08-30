#!/usr/bin/env python
#-*- coding:utf8 -*-

import MySQLdb
import json
import numpy as np
import requests
import sys
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import time
import os
from PIL import Image

# 添加第三个参数，表示从腾讯开发机（tt）拉数据还是从金山开发机（js）拉数据
urllist = sys.argv[1]
image_path = sys.argv[2]
which_db = sys.argv[3]
rename_md5 = sys.argv[4] # 采用md5进行重命名，当执行数据修正时，需要保留原始文件名（文件名中保留了原始目录信息）
assert(which_db == 'tt' or which_db == 'js')
md5_check = False # True，执行md5检查和插入
#rename_md5 = True # 采用md5进行重命名，当执行数据修正时，需要保留原始文件名（文件名中保留了原始目录信息）
if rename_md5:
  remove_duplicate = True # 删除重复的图片
else:
  remove_duplicate = False

def dict_build_up():
  cur.execute("select md5,path from md5_lib")
  a = cur.fetchall()
  md5_dict = {}

  for line in a:
      md5 = line[0]
      path = line[1].strip()
      md5_dict[md5] = path
  return md5_dict

def download(url, md5_check, rename_md5):
  try:
    pic = requests.get(url, timeout = 100)
    fmd5 = hashlib.md5(pic.content).hexdigest()
    if (pic.status_code != 200):
      print('error download pic')
      pic.raise_for_status()
    elif (pic.headers['Content-Type'] not in ['image/jpeg','image_jpeg',
        'image/png', 'image_png']):
      print('download pic header is not jpg/png')
      raise TypeError
    elif (remove_duplicate and (fmd5 in md5_dict)):
      print('repeat' + '\t' + url + "\t" + md5_dict[fmd5])
    else:
      md5_dict[fmd5] = url
      label = image_path.split('/')[-1].strip()
      line = "insert into md5_lib values ('" + str(fmd5) + "'" + ",1,'" + "/data/project/xing/new_data_set/" + image_path + "/" + str(fmd5) + ".jpg','" + label +  "')"
      insert_dict[fmd5] = line
      if(len(insert_dict) % 500 == 0):
        print(len(insert_dict),time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())))
      if rename_md5:
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

pool = ThreadPoolExecutor(max_workers=200)

if which_db == 'tt':
  conn = MySQLdb.connect(
        host='10.66.130.187',
        port = 3306,
        user='root',
        passwd='shumeiShumei2016',
        db ='data',
        )
else:
  conn = MySQLdb.connect(
        host='10.0.1.138',
        port = 3306,
        user='admin',
        passwd='shumeiShumei2016',
        db ='data',
        )
cur = conn.cursor()

future_list = []
if md5_check:
  md5_dict = dict_build_up()
else:
  md5_dict = {}
insert_dict = {}
urls = []
with open(urllist, "r") as f:
  count=0
  for line in f:
    count += 1
    line = line.strip()
    urls.append(line)

for data in urls:
  future1 = pool.submit(download, data, md5_check, rename_md5)
  future_list.append(future1)

wait(future_list)
if md5_check:
  for md5 in insert_dict:
    cur.execute(insert_dict[md5])
    conn.commit()
