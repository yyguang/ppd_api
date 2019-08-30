# coding=utf-8
# 同步图片检测服务接口, 会实时返回检测的结果

from aliyunsdkcore import client
from aliyunsdkcore.profile import region_provider
from aliyunsdkgreen.request.v20170112 import ImageSyncScanRequest
import json
import uuid
import datetime
import sys
import ConfigParser
import logging
import traceback

logging.basicConfig(level=logging.DEBUG,filename=sys.argv[3],filemode='a',format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
cf = ConfigParser.ConfigParser()
cf.read("aliyun.ak.conf")
# 请替换成你自己的accessKeyId、accessKeySecret, 您可以类似的配置在配置文件里面，也可以直接明文替换
clt = client.AcsClient(cf.get("AK", "accessKeyId"), cf.get("AK", "accessKeySecret"),'cn-shanghai')
region_provider.modify_point('Green', 'cn-shanghai', 'green.cn-shanghai.aliyuncs.com')
request = ImageSyncScanRequest.ImageSyncScanRequest()
request.set_accept_format('JSON')
delimiter_dict={'0':'\t','1':' '}
label_dict = {'normal': 0, 'porn': 1, 'sexy': 2}
def task(url):
    label=''
    rate=0.0
    # 同步现支持单张图片，即一个task
    task1 = {"dataId": str(uuid.uuid1()),
             "url":url,
             "time":datetime.datetime.now().microsecond
            }

    request.set_content(bytearray(json.dumps({"tasks": [task1], "scenes": ["porn"]}), "utf-8"))

    response = clt.do_action(request)
    result = json.loads(response)
    logging.info(result)
    if 200 == result["code"]:
        taskResults = result["data"]
        for taskResult in taskResults:
            if (200 == taskResult["code"]):
                sceneResults = taskResult["results"]
                for sceneResult in sceneResults:
                    label = sceneResult["label"]
                    rate = sceneResult["rate"]
    result = label_dict[label]
    rates=[-1]*len(label_dict)
    rates[result]=rate
    return url,result,rates[0],rates[1],rates[2]
if __name__ == '__main__':
    if len(sys.argv)!=4:
        print("Usage: python ali_api.py image_list_path delimiter log_file_path")
        exit(1)
    image_dir_name = sys.argv[1]
    delimiter = delimiter_dict[sys.argv[2]]
    print(len(label_dict))
    label_dict_sorted = [v for v in sorted(label_dict.values())]
    new_dict = {v: k for k, v in label_dict.items()}
    for v in label_dict_sorted:
        print('%d %s' % (v, new_dict[v]))
    with open(image_dir_name,'rb') as dir_list:
        for image in dir_list:
            url = image.split(delimiter)[0].strip()
            ground_truth = label_dict[image.split(delimiter)[1].strip()]
            if image.startswith("."):
                continue
            for i in range(3):
                try:
                    url, result, normal_score, porn_score, sexy_score = task(url)
                    final = '%s\t%d\t%f\t%f\t%f\t%d' % (url, result, normal_score, porn_score, sexy_score,ground_truth)
                    print(final)
                except Exception as e:
                    traceback.print_exc(e)
                break
