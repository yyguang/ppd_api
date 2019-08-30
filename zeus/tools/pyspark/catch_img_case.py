# -*- coding:UTF-8 -*

from __future__ import print_function
from pyspark import SparkContext, SparkConf
import json
import sys
import os

md5_set = set()

porn_labels = {0: "normal", 1: "porn", 2: "sexy"}
porn_labels_4 = {0: "normal", 1: "porn", 2: "exposed", 3: "sexy"}
porn_labels_7 = {0: "exposed", 1: "naked", 2: "normal", 3: "porn", 4: "sex", 5: "sexy", 6: "sm"}

def data_preprocess(line, org_obj):
  data_dict = json.loads(line.strip())
  result = ""
  try:
    org = data_dict["organization"]
    timestamp = data_dict["timestamp"]
    if org in org_obj:
      ts = data_dict["timestamp"]
      features = data_dict["features"]
      channel = features["data.channel"]
      #type = data_dict["type"]
      img_url = features.get("data.imgFileUrl", "")
      if not img_url:
        img_url = features.get("data.imgUrlFile", "")
      if not img_url:
        img_url = features.get("data.imgUrl", "")
      #riskLevel = features["rule-engine.riskLevel"]
      #riskType = features["rule-engine.riskType"]
      #text = features["imgdetector.text.text"]
      #description = features["rule-engine.description"]
      model = features["rule-engine.model"]
      #request_id = data_dict["requestId"]
      tokenId = features["data.tokenId"]
      
      normal_rate = features["img-processor-porn.porn_recognition.result.normal"]
      porn_rate = features["img-processor-porn.porn_recognition.result.porn"]
      sexy_rate = features["img-processor-porn.porn_recognition.result.sexy"]
      '''
      normal_rate_4 = features["img-processor-porn.4classes_porn_recognition.result.normal"]
      porn_rate_4 = features["img-processor-porn.4classes_porn_recognition.result.porn"]
      serious_porn_rate = features["img-processor-porn.4classes_porn_recognition.result.serious_porn"]
      sexy_rate_4 = features["img-processor-porn.4classes_porn_recognition.result.sexy"]
      '''
      #tupu_review = features.get("imgdetector-1500.baidu_porn.tupu_porn.porn.review", "")
      #tupu_label = features.get("imgdetector-1500.baidu_porn.tupu_porn.porn.label", "")
      #tupu_rate = features.get("imgdetector-1500.baidu_porn.tupu_porn.porn.rate", "")
      md5 = features["data.origin_md5"]
      if img_url != "" and md5 not in md5_set:
        md5_set.add(md5)
        rates = [normal_rate, porn_rate, sexy_rate]
        #rates_4 = [normal_rate_4, porn_rate_4, serious_porn_rate, sexy_rate_4]
        label = porn_labels[rates.index(max(rates))]
        #label_4 = porn_labels_4[rates_4.index(max(rates_4))]
       
        #result = img_url
        #if 0.8 >= porn_rate >= 0.5:
        result = "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(img_url, normal_rate, porn_rate, sexy_rate, org, label, tokenId)
        #result = "{}\t{}".format(img_url, label)
        #if riskType == 300 and riskLevel == "REJECT" and text != "":
        #   result = "{}\t{}".format(img_url, text)
  except Exception as e:
    print("{}:{}".format(e,line))
  return result


if __name__=="__main__": 
  #if len(sys.argv)!=3:
  #  print("Usage: {} <LOGIN|REGISTER> <date>".format(argv[0]))
  #label = sys.argv[1]
  dts = sys.argv[1]
  output = sys.argv[2]
  org_obj = sys.argv[3]
  if "," in org_obj:
    org_obj = set(org_obj.split(","))
  else:
    org_obj = set([org_obj])
  #porn_rate_threshold = float(sys.argv[4])
  filepaths = []
  #for dt in dts.split(","):
  if "," in dts:
    start, end = dts.split(",")
    for dt in range(int(start), int(end)+1):
      filepaths.append("/user/data/event/detail_ae/dt={}/serviceId=POST_IMG/".format(dt))
  else:
    filepaths.append("/user/data/event/detail_ae/dt={}/serviceId=POST_IMG/".format(dts))
  conf = SparkConf().setAppName("CatchImgData")
  sc = SparkContext(conf=conf)
  data = sc.textFile(",".join(filepaths))
  newRDD = data.map(lambda line:data_preprocess(line, org_obj)).filter(lambda x: x!="")
  newRDD.repartition(1).saveAsTextFile("result/{}/".format(output))



