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
	StructField("img-processor-porn.general_ad_hwad_recognition.result.dianshang",DoubleType(),True),
	StructField("img-processor-porn.general_ad_hwad_recognition.result.jupaizi",DoubleType(),True),
	StructField("img-processor-porn.general_ad_hwad_recognition.result.shouxiedazi",DoubleType(),True),
	StructField("img-processor-porn.general_ad_hwad_recognition.result.shouxieti",DoubleType(),True),
	StructField("img-processor-porn.general_ad_hwad_recognition.result.weixin",DoubleType(),True),
	StructField("img-processor-porn.general_ad_hwad_recognition.result.yinbishuiyin",DoubleType(),True),
    StructField("img-processor-porn.general_ad_hwad_recognition.result.zhengchang", DoubleType(), True),
	StructField("img-processor-porn.general_ad_hwad_recognition.label",StringType(),True),
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
  features.`img-processor-porn.general_ad_hwad_recognition.result.dianshang` as shumei_dianshang_rate,
  features.`img-processor-porn.general_ad_hwad_recognition.result.jupaizi` as shumei_jupaizi_rate,
  features.`img-processor-porn.general_ad_hwad_recognition.result.shouxiedazi` as shumei_shouxiedazi_rate,
  features.`img-processor-porn.general_ad_hwad_recognition.result.shouxieti` as shumei_shouxieti_rate,
  features.`img-processor-porn.general_ad_hwad_recognition.result.weixin` as shumei_weixin_rate,
  features.`img-processor-porn.general_ad_hwad_recognition.result.yinbishuiyin` as shumei_lianxifangshishuiyin_rate,
  features.`img-processor-porn.general_ad_hwad_recognition.result.zhengchang` as shumei_zhengchang_rate,
  features.`img-processor-porn.general_ad_hwad_recognition.label` as shumei_general_ad,
  features.`rule-engine.description` as shumei_rule_engine_result,
  features.`data.text` as text,
  features.`data.origin_md5` as md5
  from t_img_ae_baidu
  group by
  requestId,
  organization,
  data.imgUrl,
  data.imgUrlFile,
  data.imgFileUrl,
  features.`img-processor-porn.general_ad_hwad_recognition.result.dianshang`,
  features.`img-processor-porn.general_ad_hwad_recognition.result.jupaizi`,
  features.`img-processor-porn.general_ad_hwad_recognition.result.shouxiedazi`,
  features.`img-processor-porn.general_ad_hwad_recognition.result.shouxieti`,
  features.`img-processor-porn.general_ad_hwad_recognition.result.weixin`,
  features.`img-processor-porn.general_ad_hwad_recognition.result.yinbishuiyin`,
  features.`img-processor-porn.general_ad_hwad_recognition.result.zhengchang`,
  features.`img-processor-porn.general_ad_hwad_recognition.label`,
  features.`rule-engine.description`,
  features.`data.text`,
  features.`data.origin_md5`
  """)


#save_path_shumei = "/user/zhoushengyao/diff/" + date1 + "/shumei_all"
#df_result.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_shumei)
df_result.registerTempTable("t_img_ae_shumei_all")

def save_data(data_key):
  key_list = ['shumei_dianshang_rate','shumei_jupaizi_rate','shumei_shouxiedazi_rate','shumei_shouxieti_rate','shumei_weixin_rate','shumei_lianxifangshishuiyin_rate','shumei_zhengchang_rate']
  line = " select requestId,organization,imgUrlFile," + str(data_key) + ",md5 from  t_img_ae_shumei_all where shumei_general_ad!='zhengchang' and "+str(data_key) + ">=" + str(key_list[0]) +" and "+ str(data_key) + ">=" +str(key_list[1])+" and "+ str(data_key) + ">=" +str(key_list[2]) \
				         +" and "+ str(data_key) + ">=" +str(key_list[3]) \
				      +" and "+ str(data_key) + ">=" +str(key_list[4]) \
				      +" and "+ str(data_key) + ">=" +str(key_list[5]) \
				      +" and "+ str(data_key) + ">=" +str(key_list[6])


  df_result_terror = sqlContext.sql(line)
  save_path_shumei = "diff/" + date1 + '/ad/' + data_key.replace('_rate','')
  helper.check_hdfs(save_path_shumei)
  
  df_result_terror.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_shumei)

smoke_list = ['shumei_dianshang_rate','shumei_jupaizi_rate','shumei_shouxiedazi_rate','shumei_shouxieti_rate','shumei_weixin_rate','shumei_lianxifangshishuiyin_rate']
for key in smoke_list:
  save_data(key)
'''
df_result_terror = sqlContext.sql(""" select requestId,organization,imgUrlFile,tupu_terror_label,tupu_terror_rate, md5 from t_img_ae_shumei_all where tupu_terror_label != 0""")
save_path_shumei = "/user/fujunhao/diff/" + date1 + "/tupu_terror"
df_result_terror.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_shumei)
'''
