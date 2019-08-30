#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""易盾图片在线检测接口python示例代码
接口文档: http://dun.163.com/api.html
python版本：python2.7
运行:
    1. 修改 SECRET_ID,SECRET_KEY,BUSINESS_ID 为对应申请到的值
    2. $ python image_check_api_demo.py
"""
__author__ = 'yidun-dev'
__date__ = '2016/3/10'
__version__ = '0.1-dev'

import hashlib
import time
import random
import urllib
import urllib2
import json
import base64
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def image2base64(path):
  with open(path,'rb')as f:
    img64 = base64.b64encode(f.read())
  return img64

class ImageCheckAPIDemo(object):
    """图片在线检测接口示例代码"""

    API_URL = "https://as.dun.163yun.com/v3/image/check"
    VERSION = "v3.2"

    def __init__(self, secret_id, secret_key, business_id):
        """
        Args:
            secret_id (str) 产品密钥ID，产品标识
            secret_key (str) 产品私有密钥，服务端生成签名信息使用
            business_id (str) 业务ID，易盾根据产品业务特点分配
        """
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.business_id = business_id

    def gen_signature(self, params=None):
        """生成签名信息
        Args:
            params (object) 请求参数
        Returns:
            参数签名md5值
        """
        buff = ""
        for k in sorted(params.keys()):
            buff += str(k)+ str(params[k])
        buff += self.secret_key
        return hashlib.md5(buff).hexdigest()

    def check(self, params):
        """请求易盾接口
        Args:
            params (object) 请求参数
        Returns:
            请求结果，json格式
        """
        params["secretId"] = self.secret_id
        params["businessId"] = self.business_id
        params["version"] = self.VERSION
        params["timestamp"] = int(time.time() * 1000)
        params["nonce"] = int(random.random()*100000000)
        params["signature"] = self.gen_signature(params)

        # print json.dumps(params)
        try:
            params = urllib.urlencode(params)
            request = urllib2.Request(self.API_URL, params)
            content = urllib2.urlopen(request, timeout=10).read()
            with open(sys.argv[2], 'a')as f:
              f.write(content+'\n')
              f.flush()
            #print content
            # content = "{\"code\":200,\"msg\":\"ok\",\"timestamp\":1453793733515,\"nonce\":1524585,\"signature\":\"630afd9e389e68418bb10bc6d6522330\",\"result\":[{\"image\":\"http://img1.cache.netease.com/xxx1.jpg\",\"labels\":[]},{\"image\":\"http://img1.cache.netease.com/xxx2.jpg\",\"labels\":[{\"label\":100,\"level\":2,\"rate\":0.99},{\"label\":200,\"level\":1,\"rate\":0.5}]},{\"image\":\"http://img1.cache.netease.com/xxx3.jpg\",\"labels\":[{\"label\":200,\"level\":1,\"rate\":0.5}]}]}";
            return json.loads(content)
        except Exception, ex:
            print "调用API接口失败:", str(ex)

if __name__ == "__main__":
    """示例代码入口"""
    SECRET_ID = "6b1b95a4ce0309f01ea6121d4601a5e3" # 产品密钥ID，产品标识
    SECRET_KEY = "8fa470b9b3355699a7068e447d9d22d3" # 产品私有密钥，服务端生成签名信息使用，请严格保管，避免泄露
    BUSINESS_ID = "5ba2fd8ac331ad3ca6c6424b9d25638b" # 业务ID，易盾根据产品业务特点分配
    image_check_api = ImageCheckAPIDemo(SECRET_ID, SECRET_KEY, BUSINESS_ID)

    images = []
    with open(sys.argv[1], 'r')as f:
      i = 0
      for line_ in f:
        i += 1
        line = line_.strip()
        imagebase64 = {
          "name":"{\"imageId\": "+str(line)+", \"contentId\": 78978}",
          "type":2,
          "data":image2base64(line)
        }
        images.append(imagebase64)
    images_ = []
    for i in  range(0, len(images), 32):
      images_ = images[i:i+32]
        #print json.dumps(images)
      params = {
          "images": json.dumps(images_),
          "account": "python@163.com",
          "ip": "123.115.77.137"
        }
      ret = image_check_api.check(params)
      if ret["code"] == 200:
          results = ret["result"]
          for result in results:
              #print("taskId=%s，status=%s，name=%s，labels：" %(result["taskId"],result["status"],result["name"]))
              maxLevel = -1
              name = result["name"]
              faceNames = result["details"]["faceNames"]
              for labelObj in result["labels"]:
                  label = labelObj["label"]
                  level = labelObj["level"]
                  rate  = labelObj["rate"]
                  face = ','.join(faceNames)
                  print("image:%s, label:%s, level=%s, rate=%s, faceNames=%s" %(name, label, level, rate, face))
                  maxLevel =level if level > maxLevel else maxLevel
              if maxLevel==0:
                  print "#图片机器检测结果：最高等级为\"正常\"\n"
              elif maxLevel==1:
                  print "#图片机器检测结果：最高等级为\"嫌疑\"\n"
              elif maxLevel==2:
                  print "#图片机器检测结果：最高等级为\"确定\"\n"    
            
      else:
          print "ERROR: ret.code=%s, ret.msg=%s" % (ret["code"], ret["msg"])
