#!/usr/bin/env python
#-*- coding:utf8 -*-

import MySQLdb
import json
import sys

# 添加第二个参数，表示从腾讯开发机（tt）拉数据还是从金山开发机（js）拉数据，
list_path = sys.argv[1]
which_db = sys.argv[2]
assert(which_db == 'tt' or which_db == 'js')

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
with open(list_path,'r') as f:
    for line in f:
	a = line.replace("[","(").replace("]",")")
	#print a

#print a
cur.execute("select data,labels from sentry_annotation_result where taskId in" + str(a))
res = cur.fetchall()
print "select data,labels from sentry_annotation_result where taskId in" + str(a)
print len(res)
for row in res:
  #print res
  a = json.loads(row[0])
  b = row[1]
  label = b.strip('"').strip('[').strip(']').split(",")[0].strip('"')
  line = a["url"].replace("\\","") + "\t" + label
  print(line.encode('utf-8'))
