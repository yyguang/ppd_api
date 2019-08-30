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

urllist = sys.argv[1]
image_path = sys.argv[2]
task_type = sys.argv[3]

task_table = {"ad":["md5_history_test","ad7"],\
              "politics":["md5_history_test,""politics"], \
              "porn":["history_md5_list","porn"], \
              "smoke":["md5_history_test","smoke"], \
              "violence":["md5_history_test","violence"] }

headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36', 'Referer': 'https://www.fengkongcloud.com/Event/history?serviceId=POST_IMG&appId=###'}

def dict_build_up(task_type):
  cur.execute("select md5 from "+str(task_table[task_type][0])+"  where data_type='"+str(task_table[task_type][1])+"'")
  a = cur.fetchall()
  md5_dict = {}

  for line in a:
      md5 = line[0]
      md5_dict[md5] = 'history'
  return md5_dict


def download(url):
  task_type = url[1]
  try:
    pic= requests.get(url[0], timeout = 100, headers=headers)
    fmd5=hashlib.md5(pic.content).hexdigest()
    if(pic.status_code != 200):
      pic.raise_for_status()
    elif(pic.headers['Content-Type'] not in ['image/jpeg','image_jpeg','image/png','image/webp','image/x-ms-bmp','jpg','image/jpg','application/octet-stream']):
      raise TypeError
    elif(fmd5 in md5_dict):
      print('repeat' + '\t' + url[0] + "\t" + md5_dict[fmd5])
    else:
      md5_dict[fmd5] = url[0]
      label = image_path.split('/')[-1].strip()
      line = "insert into "+str(task_table[task_type][0])+"(md5,data_type)  values ('" + str(fmd5) +  "', '"+str(task_table[task_type][1])+"')"
      insert_dict[fmd5] = line
      if(len(insert_dict) % 500 == 0):
        print(len(insert_dict),time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())))
      name = image_path + '/' + str(fmd5) + '.jpg'
      fp = open(name, 'wb')
      fp.write(pic.content)
      fp.close()
      try:
        Image.open(name).load()
      except Exception as e:
        print(str(e) + "\t" + url[0])
        os.remove(name)

  except Exception as e:
    print(str(e) + "\t" + url[0])

pool = ThreadPoolExecutor(max_workers=20)

conn= MySQLdb.connect(
        host='10.66.130.187',
        port = 3306,
        user='root',
        passwd='shumeiShumei2016',
        db ='data',
        )
cur = conn.cursor()

future_list = []
md5_dict = dict_build_up(task_type)
insert_dict = {}
urls = []
with open(urllist, "r") as f:
  count=0
  for line in f:
    count += 1
    line = line.strip()
    urls.append([line,task_type])

for data in urls:
  future1 = pool.submit(download, (data))
  future_list.append(future1)

wait(future_list)
for md5 in insert_dict:
  cur.execute(insert_dict[md5])
  conn.commit()
