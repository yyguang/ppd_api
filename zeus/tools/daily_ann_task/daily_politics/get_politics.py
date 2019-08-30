# -*- coding: UTF-8 -*-
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import sys
import json
date1 = sys.argv[1]
date1 = str(date1)
time_tag = 'I' + date1[4: 8]

IMGAE_SIMP_SCHEMA_BAIDU = StructType([
  StructField("requestId", StringType(), True),
  StructField("organization", StringType(), True),
  StructField("data", StructType([
    StructField("imgUrl", StringType(), True),
    StructField("imgUrlFile", StringType(), True),
	StructField("imgFileUrl",StringType(),True),
    StructField("tokenId", StringType(), True),
    StructField("channel", StringType(), True)
  ]), True),
  StructField("features", StructType([
	StructField("img-processor-porn.politics_recognition.result.distance",DoubleType(),True),
	StructField("img-processor-porn.politics_recognition.result.face_num",IntegerType(),True),
	StructField("img-processor-porn.politics_recognition.result.face_id",StringType(),True),
    StructField("rule-engine.description", StringType(), True),
    StructField("data.text", StringType(), True),
    StructField("data.origin_md5", StringType(), True)
  ]), True)])

conf = SparkConf().setAppName("temp_select")
sc = SparkContext(conf=conf)

sqlContext = SQLContext(sc)
log_path = "/user/data/event/detail_ae/dt=" + date1 + "/serviceId=POST_IMG/POST_IMG#part*.gz"

sqlContext.read.schema(IMGAE_SIMP_SCHEMA_BAIDU).format("json").load(log_path).registerTempTable("t_img_ae_baidu")

df_result = sqlContext.sql(""" select
  requestId,
  organization,
  data.`imgFileUrl` as imgUrlFile,
  features.`img-processor-porn.politics_recognition.result.distance` as shumei_distance_rate,
  features.`img-processor-porn.politics_recognition.result.face_num` as shumei_facenum_rate,
  features.`img-processor-porn.politics_recognition.result.face_id` as shumei_faceid_rate,
  features.`rule-engine.description` as shumei_rule_engine_result,
  features.`data.text` as text,
  features.`data.origin_md5` as md5
  from t_img_ae_baidu where features.`img-processor-porn.politics_recognition.result.face_id` in ('金正恩','彭丽媛','江泽民') and \
  features.`img-processor-porn.politics_recognition.result.distance` <= 0.12 group by \
  requestId,
  organization,
  data.imgUrl,
  data.imgUrlFile,
  data.imgFileUrl,
  features.`img-processor-porn.politics_recognition.result.face_num`,
  features.`img-processor-porn.politics_recognition.result.face_id`,
  features.`img-processor-porn.politics_recognition.result.distance`,
  features.`rule-engine.description`,
  features.`data.text`,
  features.`data.origin_md5`
  """)


df_result.registerTempTable("t_img_ae_shumei_all")

def save_data():
  line = " select requestId,organization,imgUrlFile,shumei_distance_rate,shumei_facenum_rate,shumei_faceid_rate,md5 from  t_img_ae_shumei_all limit 500"
  df_result_terror = sqlContext.sql(line)
  save_path_shumei = "diff/" + date1 + '/politics/'
  df_result_terror.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_shumei)

save_data()
