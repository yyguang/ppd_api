# -*- coding: utf-8 -*-
import requests
import base64
import hashlib
import os
import json
import sys
import time
label_dict = {'normal': 0, 'porn': 1, 'sexy': 2}
label_dict2 = {'正常'.decode('utf-8'): 0, '色情'.decode('utf-8'): 1, '性感'.decode('utf-8'): 2} 


def shumei_image_detect(image, timeout):
  image_handler = open(image, "rb")
  image_byte = image_handler.read()
  image_base64 =base64.b64encode(image_byte)
  scenes = ["antiporn"]
  data = {"image": image_base64, "scenes": scenes}
  headers = {"Content-Type": "application/json;charset=utf-8"}
  data = json.dumps(data)
  shumei_url = "https://aip.baidubce.com/api/v1/solution/direct/img_censor?access_token=24.124aa2a1aef6fea91b37680b49f47d12.2592000.1522462836.282335-9344323"
  shumei_result = requests.post(shumei_url, headers=headers, data=data, timeout=timeout)
  encode_result = json.loads(shumei_result.text)
  normal_score = encode_result['result']['antiporn']['result'][2]['probability']
  porn_score = encode_result['result']['antiporn']['result'][0]['probability']
  sexy_score = encode_result['result']['antiporn']['result'][1]['probability']
  result = encode_result['result']['antiporn']['conclusion']
  result = label_dict2[result]
  return normal_score, porn_score, sexy_score, result 

def shumei_dir_list(dir_path, timeout):
  print(len(label_dict))
  for key, item in label_dict.items():
    print('%d %s' % (item, key))
  with open(dir_path, 'r') as dir_list:
    for image in dir_list:
     url = image.split('\t')[0].strip()
     ground_truth = label_dict[image.split('\t')[1].strip()]
     if image.startswith("."):
       continue
     for i in range(3):
       try:
         normal_score, porn_score, sexy_score, result = shumei_image_detect(url, timeout)
         final = '%s\t%d\t%f\t%f\t%f\t%d' % (url, result, normal_score, porn_score, sexy_score, ground_truth)
         print(final)
       except:
         continue
       break

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python get_result_from_baidu_api.py image_list_path")
    exit(1)
  request_timeout = 10
  image_dir_name = sys.argv[1]
  shumei_dir_list(image_dir_name, request_timeout)
