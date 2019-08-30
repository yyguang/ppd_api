#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author: chenyunkuo

#coding=utf-8
import MySQLdb
import json
import sys
import re
#start = sys.argv[1]
#end = sys.argv[2]

start, end = sys.argv[1].split(",")

conn= MySQLdb.connect(
        host='10.0.0.20',
        port = 3306,
        user='saas',
        passwd='shumeiShumei2018',
        db ='data',
        )
cur = conn.cursor()
line = "select taskId, data, labels from sentry_annotation_result where taskId <= {} and taskId >= {};".format(end, start)
#line = "select taskId, data, labels from sentry_annotation_result where taskId in ('454312','454313');"
cur.execute(line)
res = cur.fetchall()
print len(res)
#print res[0]

labeled = []
pat = re.compile("[\d|\.]+")
result = []
for row in res:
  #print row
  a = json.loads(row[1])
  if "url" in a:
   taskId = row[0]
   label = row[2]
   image = a["url"].replace("\\","")
   md5 = image.split("/")[-1]
   labeled.append(md5)
   coords = pat.findall(label)
   if len(coords) >= 4 and len(coords) % 4 == 0:
     result.append("{}\t{}".format(md5, ",".join(coords)))
     #result.append("{}\t{}\t{}".format(taskId, image, ",".join(coords)))
   #elif len(label) > 2:
   #  print(label)
  else:
    print(row)
  
with open("md5_random_crop.list", "w") as f:
  f.write("\n".join(result) + "\n")

with open("porn_crop_rectangled.list", "w") as f:
  f.write("\n".join(labeled) + "\n")
