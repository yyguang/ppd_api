# -*- coding: utf-8 -*-
# Author chenyunkuo <chenyunkuo@ishumei.com>
import requests
import base64
import hashlib
import os
import json
import sys
import time
import traceback
import logging

if os.path.exists("public.log"):
    os.remove("public.log")

logging.basicConfig(level=logging.DEBUG,
                    filename="public.log",
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
result_dict={}

def shumei_image_detect(image, timeout):
    image_handler=open(image,"rb")
    image_byte=image_handler.read()
    image_base64=base64.b64encode(image_byte)
    scenes=["public"]
    data={"image":image_base64,"scenes":scenes}
    headers = {"Content-Type": "application/json;charset=utf-8"}
    data = json.dumps(data)
    shumei_url="https://aip.baidubce.com/api/v1/solution/direct/img_censor?access_token=24.32724db248b56df4f265b0630b1dcf42.2592000.1559874759.282335-9344323"
    shumei_result = requests.post(shumei_url, headers=headers, data=data, timeout=timeout)
    encode_result = json.loads(shumei_result.text)
    logging.info(encode_result)
    public = encode_result["result"]["public"]
    #result_num
    result_num = public["result_num"]
    #stars
    stars = public["result"][0]["stars"][0]
    #star_name
    name = stars["name"]
    #probability
    prob = stars["probability"]
    #star_id
    star_id = stars["star_id"]
    return result_num, name, prob, star_id


def shumei_dir_list(dir_path, result_file, timeout):
    with open(result_file, 'w') as f:
        with open(dir_path, 'r') as dir_list:
            for image in dir_list:
                img = image.split()[0].strip()
                ground_truth = image.split()[1].strip()
                if image.startswith("."):
                    continue
                for i in range(3):
                    try:
                        result_num, name, prob, star_id = shumei_image_detect(img, timeout)
                        result = '{}\t{}\t{}\t{}\t{}\t{}'.format(img, ground_truth, name.encode('utf-8'), prob, star_id, result_num)
                        f.write(result + "\n")
                    except Exception as e:
                        traceback.print_exc(e)
                    break

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python baidu_api_publicface.py image_list_path log_file_path")
    exit(1)
  request_timeout = 10
  image_dir_name = sys.argv[1]
  result_file = sys.argv[2]
  shumei_dir_list(image_dir_name, result_file, request_timeout)
