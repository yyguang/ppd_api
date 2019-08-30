# -*- coding: UTF-8 -*-
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import sys
import json
sys.path.append("../")
import dailyTaskHelper as helper

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
    StructField("img-processor-porn.porn_recognition.result.normal", DoubleType(), True),
    StructField("img-processor-porn.porn_recognition.result.porn", DoubleType(), True),
    StructField("img-processor-porn.porn_recognition.result.sexy", DoubleType(), True),
    StructField("img-processor-porn.terror_recognition.result.result.kongbuzuzhi", DoubleType(), True),
    StructField("img-processor-porn.terror_recognition.result.result.baoluanchangjing", DoubleType(), True),
    StructField("img-processor-porn.terror_recognition.result.result.guoqiguohui", DoubleType(), True),
    StructField("img-processor-porn.terror_recognition.result.result.junzhuang", DoubleType(), True),
    StructField("img-processor-porn.terror_recognition.result.result.qiangzhidaoju", DoubleType(), True),
    StructField("img-processor-porn.terror_recognition.result.result.zhengchangzongjiao", DoubleType(), True),
    StructField("img-processor-porn.terror_recognition.result.result.youxiqiangzhidaoju", DoubleType(), True),
    StructField("img-processor-porn.terror_recognition.result.result.ertongxiedian", DoubleType(), True),
    StructField("img-processor-porn.terror_recognition.result.result.xuexingchangjing", DoubleType(), True),
	StructField("img-processor-porn.behavior_recognition.result.smoke",DoubleType(),True),
    StructField("imgdetector.nmviolence.label", DoubleType(), True),
	StructField("imgdetector.tupuscence.rate",DoubleType(),True),
	StructField("imgdetector.tupuscence.label",IntegerType(),True),
    StructField("imgdetector.nmviolence.rate", DoubleType(), True),
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
  features.`img-processor-porn.behavior_recognition.result.smoke` as shumei_smoke_rate,
  features.`imgdetector.tupuscence.rate` as tupu_smoke_rate,
  features.`imgdetector.tupuscence.label` as tupu_smoke_label,
  features.`rule-engine.description` as shumei_rule_engine_result,
  features.`data.text` as text,
  features.`data.origin_md5` as md5
  from t_img_ae_baidu where organization in ('MEnxzBzYYTJqEfrXN4oS','Zjf083WlTqlJpwgU0q9U') 
  group by
  requestId,
  organization,
  data.imgUrl,
  data.imgUrlFile,
  data.imgFileUrl,
  features.`img-processor-porn.behavior_recognition.result.smoke`,
  features.`imgdetector.tupuscence.rate`,
  features.`imgdetector.tupuscence.label`,
  features.`rule-engine.description`,
  features.`data.text`,
  features.`data.origin_md5`
  """)


#save_path_shumei = "/user/zhoushengyao/diff/" + date1 + "/shumei_all"
#df_result.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_shumei)
df_result.registerTempTable("t_img_ae_shumei_all")

def save_data(data_key, threshold):
  if data_key == 'tupu_smoke_rate':
    line = ' select requestId,organization,imgUrlFile,' + str(data_key) + \
    ',md5 from  t_img_ae_shumei_all where tupu_smoke_label  = 4 and ' + str(data_key) + ' > ' + str(threshold)
  else:
    line = ' select requestId,organization,imgUrlFile,' + str(data_key) + \
		       ',md5 from  t_img_ae_shumei_all where ' + str(data_key) + ' > ' + str(threshold)
  df_result_terror = sqlContext.sql(line)
  save_path_shumei = "diff/" + date1 + '/smoke/' + data_key.replace('_rate','')
  helper.check_hdfs(save_path_shumei)
  df_result_terror.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_shumei)

smoke_list = ['shumei_smoke_rate']
for key in smoke_list:
  save_data(key,'0.5')
'''
df_result_terror = sqlContext.sql(""" select requestId,organization,imgUrlFile,tupu_terror_label,tupu_terror_rate, md5 from t_img_ae_shumei_all where tupu_terror_label != 0""")
save_path_shumei = "/user/fujunhao/diff/" + date1 + "/tupu_terror"
df_result_terror.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_shumei)
'''
