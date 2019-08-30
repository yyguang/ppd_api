# -*- coding: utf-8 -*-
# Author fujunhao <fujunhao@ishumei.com>
import requests
import base64
import hashlib
import os
import json
import sys
import time
import traceback
import logging

logging.basicConfig(level=logging.DEBUG,
                    filename=sys.argv[3],
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
result_dict={}
delimiter_dict={'0':'\t','1':' '}

def shumei_image_detect(image,timeout):
    image_handler=open(image,"rb")
    image_byte=image_handler.read()
    image_base64=base64.b64encode(image_byte)
    scenes=["politician"]
    data={"image":image_base64,"scenes":scenes}
    headers = {"Content-Type": "application/json;charset=utf-8"}
    data = json.dumps(data)
    shumei_url="https://aip.baidubce.com/api/v1/solution/direct/img_censor?access_token=24.9f5e317bb09a8f534e766f8cb4f442e2.2592000.1524728084.282335-9344323"
    shumei_result = requests.post(shumei_url, headers=headers, data=data, timeout=timeout)
    encode_result = json.loads(shumei_result.text)
    logging.info(encode_result)
    politics=encode_result["result"]["politician"]
    #result_num
    result_num = politics["result_num"]
    #include_politician
    include_politician = politics["include_politician"]
    #result_confidence
    result_confidence = politics["result_confidence"]
    return result_num, include_politician, result_confidence


def shumei_dir_list(dir_path, timeout):
    delimiter = delimiter_dict[sys.argv[2]]
    with open(dir_path, 'r') as dir_list:
        for image in dir_list:
            url = image.split(delimiter)[0].strip()
            try:
                result_num, include_politician, result_confidence = shumei_image_detect(url, timeout)
                final = '%s\t%d\t%s\t%s' % (url, result_num, include_politician, result_confidence)
                print(final)
            except Exception as e:
                traceback.print_exc(e)
                break

if __name__ == "__main__":
  if len(sys.argv) != 4:
    print("Usage: python baidu_api_terr.py image_list_path delimiter log_file_path")
    exit(1)
  request_timeout = 10
  image_dir_name = sys.argv[1]
  shumei_dir_list(image_dir_name, request_timeout)
