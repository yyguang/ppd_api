#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author fu jun hao <fujunhao@@ishumei.com>
from __future__ import print_function
from pyspark import SparkContext, SparkConf
import json
import sys
import os

md5_set = set()

def data_preprocess(line,special):
  data_dict = json.loads(line.strip())
  result = ""
  try:
    org = data_dict["organization"]
    features = data_dict["features"]
    img_url = features.get("data.imgFileUrl", "")
    if not img_url:
      img_url = features.get("data.imgUrlFile", "")
    if not img_url:
      img_url = features.get("data.imgUrl", "")
    md5 = features["data.img_md5"]
    if special == "violence":
      label = features["img-processor-porn.terror_recognition.result.label"]
      rate = features["img-processor-porn.terror_recognition.result.labelProbability"]
    elif special == "porn":
      label = features["img-processor-porn.porn_recognition.label"]
      rate = features["img-processor-porn.porn_recognition.labelProbability"]
    elif special == "porn4":
      label = features["img-processor-porn.4classes_porn_recognition.label"]
      rate = features["img-processor-porn.4classes_porn_recognition.labelProbability"]
    if img_url != "" and md5 not in md5_set:
      md5_set.add(md5)
      result = "{}\t{}\t{}".format(img_url, label, rate)
  except Exception as e:
    print("{}:{}".format(e,line))
  return result


if __name__=="__main__": 
  dts = sys.argv[1]
  output = sys.argv[2]
  special = sys.argv[3]
  sample_num = int(sys.argv[4])
  filepaths = []
  if "," in dts:
    start, end = dts.split(",")
    for dt in range(int(start), int(end)+1):
      filepaths.append("/user/data/event/detail_ae/dt={}/serviceId=POST_IMG/".format(dt))
  else:
    filepaths.append("/user/data/event/detail_ae/dt={}/serviceId=POST_IMG/".format(dts))
  conf = SparkConf().setAppName("CatchImgData")
  sc = SparkContext(conf=conf)
  data = sc.textFile(",".join(filepaths))
  newRDD = data.map(lambda line:data_preprocess(line, special)).filter(lambda x: x!="")
  #takeSample函数类似于sample函数，该函数接受三个参数，第一个参数withReplacement ，表示采样是否放回，true表示有放回的采样，false表示无放回采样
  sc.parallelize(newRDD.takeSample(False, sample_num)).repartition(1).saveAsTextFile("result/{}/".format(output))
