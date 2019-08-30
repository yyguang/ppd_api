# -*- coding: utf-8 -*-
import requests
import base64
import hashlib
import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait

label_dict = {'normal': 0, 'porn': 1, 'sexy': 2}
label_dict2 = {'正常'.decode('utf-8'):0, '色情'.decode('utf-8'): 1, '性感'.decode('utf-8'): 2} 

def shumei_image_detect(image, timeout):
  image_handler = open(image, "rb")
  image_byte = image_handler.read()
  image_base64 =base64.b64encode(image_byte)
  scenes = ["antiporn"]
  data = {"image": image_base64, "scenes": scenes}
  headers = {"Content-Type": "application/json;charset=utf-8"}
  data = json.dumps(data)
  shumei_url = "https://aip.baidubce.com/api/v1/solution/direct/img_censor?access_token=24.fafb92ae19e7f72ffbfcae7b1151fbd5.2592000.1525401473.282335-9344323"
  shumei_result = requests.post(shumei_url, headers=headers, data=data, timeout=timeout)
  encode_result = json.loads(shumei_result.text)
  normal_score = encode_result['result']['antiporn']['result'][2]['probability']
  porn_score = encode_result['result']['antiporn']['result'][0]['probability']
  sexy_score = encode_result['result']['antiporn']['result'][1]['probability']
  result = encode_result['result']['antiporn']['conclusion']
  result = label_dict2[result]
  return normal_score, porn_score, sexy_score, result 

def shumei_dir_list(image, timeout):
  url = image.split()[0].strip()
  ground_truth = label_dict[image.split()[1].strip()]
  for i in range(3):
    try:
      normal_score, porn_score, sexy_score, result = shumei_image_detect(url, timeout)
      final = '%s\t%d\t%f\t%f\t%f\t%d' % (url, result, normal_score, porn_score, sexy_score, ground_truth)
      result_set.append(final)
      if(len(result_set) % 500 == 0):
        print(len(result_set),time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())))
    except Exception as e:
      print(e,url,"Error")
      continue
    break

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python multi_th_get_result_from_baidu_api.py image_list_path out_put_file_path")
    exit(1)
  request_timeout = 100
  image_dir_name = sys.argv[1]
  dest_dir = sys.argv[2]
  result_set = []
  pool = ThreadPoolExecutor(max_workers=10)
  future_list = []
  with open(image_dir_name,'r') as f:
    for image in f:
      future1 = pool.submit(shumei_dir_list, image, request_timeout)
      future_list.append(future1)
  wait(future_list)
  with open(dest_dir,'wb') as f:
    f.write('%d\n' % len(label_dict))
    for key, item in label_dict.items():
      f.write('%d %s\n' % (item, key))
    for line in result_set:
      f.write(line+"\n")  
