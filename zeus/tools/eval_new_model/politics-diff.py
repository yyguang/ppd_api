#encoding:utf-8
# Author fu jun hao <fujunhao@ishumei.com>
import sys
import os
import re
import json


def read_log(filename):
  tmp = {}
  requests = []
  with open(filename,'r')as f:
    for line_ in f:
      try:
        item = {}
        line = line_.strip().split()
        requestid = line[6].split("=")[1]
        imgurl = line[8].split("=")[1]
        result = line[9].split("=")[1]
        detail = json.loads(result)
        politics_recognition = detail['politics_recognition']["result"]
        face_num = politics_recognition["face_num"]
        face_id = politics_recognition["face_id"]
        distance = politics_recognition["distance"]
        is_hit_cache_politics = politics_recognition["is_hit_cache_politics"]
        if not is_hit_cache_politics:
          tmp.setdefault(requestid,[imgurl,face_num,face_id,distance])
          requests.append(requestid)
        #print requestid,imgurl,face_num,face_id,distance
      except Exception as e:
        continue
  return tmp,requests

if __name__ == "__main__":
  online,request_online = read_log(sys.argv[1])
  strom,request_strom = read_log(sys.argv[2])
  i = 0
  for item in online:
    imgurl = online[item][0]
    facenum = int(online[item][1])
    faceid = online[item][2]
    distance = float(online[item][3])
  i = 0
  for item in request_online:
    try:
      if item in request_strom:
        imgurl = online[item][0]
        facenum_online = int(online[item][1])
        faceid_online = online[item][2]
        distance_online = float(online[item][3])
        facenum_strom = int(strom[item][1])
        faceid_strom = strom[item][2]
        distance_strom = float(strom[item][3])
        if (distance_online != distance_strom) or (faceid_strom != faceid_online):
          print imgurl,facenum_online,faceid_online,distance_online,facenum_strom,faceid_strom,distance_strom
          i += 1
      else:
        continue
        print imgurl,facenum_online,faceid_online,distance_online,facenum_strom,faceid_strom,distance_strom
        #print item,imgurl,facenum_online,faceid_online,distance_online,facenum_strom,faceid_strom,distance_strom
    except Exception as e:
      #print online[item][2]
      continue
  print "total  diff ", str(i)
