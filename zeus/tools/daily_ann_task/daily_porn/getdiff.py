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
    StructField("imgFileUrl", StringType(), True),
    StructField("tokenId", StringType(), True),
    StructField("channel", StringType(), True)
  ]), True),
  StructField("features", StructType([
    StructField("img-processor-porn.porn_recognition.result.normal", DoubleType(), True),
    StructField("img-processor-porn.porn_recognition.result.porn", DoubleType(), True),
    StructField("img-processor-porn.porn_recognition.result.sexy", DoubleType(), True),
    StructField("rule-engine.description", StringType(), True),
    StructField("data.text", StringType(), True),
    StructField("data.origin_md5", StringType(), True),
    StructField("img-processor-porn.liebao_porn_recognition.result.serious_porn", DoubleType(), True),
    StructField("img-processor-porn.liebao_porn_recognition.result.porn", DoubleType(), True)
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
  features.`img-processor-porn.porn_recognition.result.normal` as shumei_normal_rate,
  features.`img-processor-porn.porn_recognition.result.porn` as shumei_porn_rate,
  features.`img-processor-porn.porn_recognition.result.sexy` as shumei_sexy_rate,
  features.`rule-engine.description` as shumei_rule_engine_result,
  features.`data.text` as text,
  features.`data.origin_md5` as md5,
  features.`img-processor-porn.liebao_porn_recognition.result.serious_porn` as liebao_serious_porn_rate,
  features.`img-processor-porn.liebao_porn_recognition.result.porn` as liebao_porn_rate
  from t_img_ae_baidu
  group by
  requestId,
  organization,
  data.imgUrl,
  data.imgUrlFile,
  data.imgFileUrl,
  features.`img-processor-porn.porn_recognition.result.normal`,
  features.`img-processor-porn.porn_recognition.result.porn`,
  features.`img-processor-porn.porn_recognition.result.sexy`,
  features.`rule-engine.description`,
  features.`data.text`,
  features.`data.origin_md5`,
  features.`img-processor-porn.liebao_porn_recognition.result.serious_porn`,
  features.`img-processor-porn.liebao_porn_recognition.result.porn`
  """)


#save_path_shumei = "diff/" + date1 + "/shumei_all"
#df_result.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_shumei)
df_result.registerTempTable("t_img_ae_shumei_all")

def save_data(data_key, threshold):
  line = ' select requestId,organization,imgUrlFile,' + str(data_key) + \
    ',md5 from  t_img_ae_shumei_all where ' + str(data_key) + ' > ' + str(threshold)
  df_result_terror = sqlContext.sql(line)
  save_path_shumei = "diff/" + date1 + '/porn/' + data_key.replace('_rate','')
  helper.check_hdfs(save_path_shumei)
  df_result_terror.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_shumei)

#key_list = ['shumei_kongbuzuzhi_rate','shumei_baoluanchangjing_rate','shumei_guoqiguohui_rate','shumei_junzhuang_rate','shumei_qiangzhidaoju_rate','shumei_xuexingchangjing_rate','shumei_zhengchangzongjiao_rate','shumei_ertongxiedian_rate','shumei_youxiqiangzhidaoju_rate']
#for key_data in key_list:
#  save_data(key_data, '0.5')

df_result_diff = sqlContext.sql(""" 
  select * 
  from t_img_ae_shumei_all 
  where shumei_porn_rate > shumei_sexy_rate 
  and shumei_porn_rate  > shumei_normal_rate 
  and (ISNULL(md5)) = false """)

save_path_baidu = "diff/" + date1 + "/porn/shumei_porn"
helper.check_hdfs(save_path_baidu)
df_result_diff.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_baidu)

df_result_diff = sqlContext.sql("""
  select *
  from t_img_ae_shumei_all
  where shumei_normal_rate > shumei_sexy_rate
  and shumei_normal_rate  > shumei_porn_rate
  and (ISNULL(md5)) = false """)

save_path_baidu = "diff/" + date1 + "/porn/shumei_normal"
helper.check_hdfs(save_path_baidu)
df_result_diff.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_baidu)

df_result_diff = sqlContext.sql("""
  select *
  from t_img_ae_shumei_all
  where shumei_sexy_rate > shumei_normal_rate
  and shumei_sexy_rate  > shumei_porn_rate
  and (ISNULL(md5)) = false """)
save_path_baidu = "diff/" + date1 + "/porn/shumei_sexy"
helper.check_hdfs(save_path_baidu)
df_result_diff.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_baidu)


