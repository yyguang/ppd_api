# -*- coding: utf-8 -*-
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
label_dict={"baoluanchangjing":0,"guoqiguohui":1,"junzhuang":2,"kongbuzuzhi":3,"qiangzhidaoju":4,"xuexingchangjing":5,"zhengchang":6}
label_dict2={"normal":0,"polic_army":1,
             "body":2,"explosion_fire":3,
             "killing":4,"violent":5,
             "violent_people":6,"weapons":7,
             "terror_banner":8,"bloody_animal_corpse":9}
result_dict={}
delimiter_dict={'0':'\t','1':' '}

def shumei_image_detect(image,timeout):
    image_handler=open(image,"rb")
    image_byte=image_handler.read()
    image_base64=base64.b64encode(image_byte)
    scenes=["terror"]
    data={"image":image_base64,"scenes":scenes}
    headers = {"Content-Type": "application/json;charset=utf-8"}
    data = json.dumps(data)
    shumei_url="https://aip.baidubce.com/api/v1/solution/direct/img_censor?access_token=24.9f5e317bb09a8f534e766f8cb4f442e2.2592000.1524728084.282335-9344323"
    shumei_result = requests.post(shumei_url, headers=headers, data=data, timeout=timeout)
    encode_result = json.loads(shumei_result.text)
    logging.info(encode_result)
    terror=encode_result["result"]["terror"]
    #正常
    normal_score=terror["result_fine"][0]["score"]
    result_dict['normal']=normal_score
    #警察部队
    polic_army_score=terror["result_fine"][1]["score"]
    result_dict['polic_army']=polic_army_score
    #尸体
    body_score=terror["result_fine"][2]["score"]
    result_dict['body']=body_score
    #爆炸火灾
    explosion_fire_score=terror["result_fine"][3]["score"]
    result_dict['explosion_fire']=explosion_fire_score
    #杀人
    killing_score=terror["result_fine"][4]["score"]
    result_dict['killing']=killing_score
    #暴乱
    violent_score=terror["result_fine"][5]["score"]
    result_dict['violent']=violent_score
    #暴恐人物
    violent_people_score=terror["result_fine"][6]["score"]
    result_dict['violent_people']=violent_people_score
    #军事武器
    weapons_score=terror["result_fine"][7]["score"]
    result_dict['weapons']=weapons_score
    #暴恐旗帜
    terror_banner_score=terror["result_fine"][8]["score"]
    result_dict['terror_banner']=terror_banner_score
    #血腥动物或动物尸体
    bloody_animal_corpse_score=terror["result_fine"][9]["score"]
    result_dict['bloody_animal_corpse']=bloody_animal_corpse_score
    #result
    max_score=sorted(result_dict,key=lambda x:result_dict[x])[-1]
    result=label_dict2[max_score]
    return normal_score,polic_army_score,body_score,explosion_fire_score,killing_score,violent_score,violent_people_score,weapons_score,terror_banner_score,bloody_animal_corpse_score,result


def shumei_dir_list(dir_path, timeout):
    delimiter = delimiter_dict[sys.argv[2]]
    print(len(label_dict2))
    label_dict2_sorted=[ v for v in sorted(label_dict2.values())]
    new_dict = {v: k for k, v in label_dict2.items()}
    for v in label_dict2_sorted:
        print('%d %s' % (v, new_dict[v]))
    with open(dir_path, 'r') as dir_list:
        for image in dir_list:
            url = image.split(delimiter)[0].strip()
            ground_truth = label_dict[image.split(delimiter)[1].strip()]
            if image.startswith("."):
                continue
            for i in range(3):
                try:
                    normal_score, polic_army_score, body_score, explosion_fire_score, \
                    killing_score, violent_score, violent_people_score, weapons_score, \
                    terror_banner_score, bloody_animal_corpse_score, result = shumei_image_detect(url, timeout)
                    final = '%s\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d' % (url, result, normal_score, polic_army_score, body_score, explosion_fire_score, killing_score, violent_score, violent_people_score, weapons_score, terror_banner_score, bloody_animal_corpse_score, ground_truth)
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
