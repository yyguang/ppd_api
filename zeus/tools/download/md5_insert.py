# -*- coding: UTF-8 -*-

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
import sys

#usage: python md5_insert.py md5_list data_type
def history_md5_dict_build(md5_list):
  history_md5_dict = {}
  for md5_tuple in md5_list:
    md5 = md5_tuple[0]
    history_md5_dict[md5] = 'history'
  return history_md5_dict


data_dict = {'porn':'色情','politics':'涉政','ocr':'ocr', 'violence':'暴恐','qr':'二维码','ad':'广告', 'other':'其他'}
if(len(sys.argv) != 4):
  print("usage: python md5_insert.py full_path_list data_type remove_tag")
  exit(-1)

full_path_list = sys.argv[1]
data_type = sys.argv[2]
remove_tag = bool(int(sys.argv[3]))

if(data_type not in data_dict):
  print ("data_type error ")
  print("""available type: 'porn':'色情','politics':'涉政','ocr':'ocr', 'violence':'暴恐','qr':'二维码','ad':'广告', 'other':'其他'""")
  exit(-1)

insert_dict = {}

conn= MySQLdb.connect(
        host='10.66.130.187',
        port = 3306,
        user='root',
        passwd='shumeiShumei2016',
        db ='data',
        )
cur = conn.cursor()
select_sql_content = "select md5 from history_md5_list where data_type = '" + str(data_type) + "'"
cur.execute(select_sql_content)

history_md5_list = cur.fetchall()
history_md5_dict = history_md5_dict_build(history_md5_list)


i = 0
with open(full_path_list) as f:
  for line in f:
    i += 1
    if(i % 5000 == 0):
      print str(i), time.strftime('%H:%M:%S',time.localtime(time.time()))
    line = line.strip()
    md5 = line.split('/')[-1].replace('.jpg','')
    if(md5 in history_md5_dict):
      print("repeat ",line, history_md5_dict[md5])
      if(remove_tag):
        os.remove(line)
    else:
      #print(history_md5_list)
      history_md5_dict[md5] = line 
      sql_content = "insert into history_md5_list values ('" + str(md5) +  "','" + str(data_type) + "')"
      #print sql_content
      insert_dict[md5] = sql_content

for md5 in insert_dict:
  sql_content = insert_dict[md5] 
  cur.execute(sql_content)
conn.commit()
