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
	StructField("img-processor-porn.terror_recognition.result.result.anheidongman", DoubleType(), True),
	StructField("img-processor-porn.terror_recognition.result.label",StringType(),True),
    StructField("imgdetector.nmviolence.label", DoubleType(), True),
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
  features.`img-processor-porn.terror_recognition.result.label` as shumei_normal_rate,
  features.`img-processor-porn.porn_recognition.result.porn` as shumei_porn_rate,
  features.`img-processor-porn.porn_recognition.result.sexy` as shumei_sexy_rate,
  features.`img-processor-porn.terror_recognition.result.result.kongbuzuzhi` as shumei_kongbuzuzhi_rate,
  features.`img-processor-porn.terror_recognition.result.result.baoluanchangjing` as shumei_baoluanchangjing_rate,
  features.`img-processor-porn.terror_recognition.result.result.guoqiguohui` as shumei_guoqiguohui_rate,
  features.`img-processor-porn.terror_recognition.result.result.junzhuang` as shumei_junzhuang_rate,
  features.`img-processor-porn.terror_recognition.result.result.qiangzhidaoju` as shumei_qiangzhidaoju_rate,
  features.`img-processor-porn.terror_recognition.result.result.zhengchangzongjiao` as shumei_zhengchangzongjiao_rate,
  features.`img-processor-porn.terror_recognition.result.result.xuexingchangjing` as shumei_xuexingchangjing_rate,
  features.`img-processor-porn.terror_recognition.result.result.youxiqiangzhidaoju` as shumei_youxiqiangzhidaoju_rate,
  features.`img-processor-porn.terror_recognition.result.result.ertongxiedian` as shumei_ertongxiedian_rate,
  features.`img-processor-porn.terror_recognition.result.result.anheidongman` as shumei_anheidongman_rate,
  features.`imgdetector.nmviolence.label` as tupu_terror_label,
  features.`imgdetector.nmviolence.rate` as tupu_terror_rate,
  features.`rule-engine.description` as shumei_rule_engine_result,
  features.`data.text`,
  features.`data.origin_md5` as md5
  from t_img_ae_baidu
  group by
  requestId,
  organization,
  data.imgUrl,
  data.imgUrlFile,
  data.imgFileUrl,
  features.`img-processor-porn.terror_recognition.result.label`,
  features.`img-processor-porn.porn_recognition.result.porn`,
  features.`img-processor-porn.porn_recognition.result.sexy`,
  features.`img-processor-porn.terror_recognition.result.result.kongbuzuzhi`,
  features.`img-processor-porn.terror_recognition.result.result.baoluanchangjing`,
  features.`img-processor-porn.terror_recognition.result.result.guoqiguohui`,
  features.`img-processor-porn.terror_recognition.result.result.junzhuang`,
  features.`img-processor-porn.terror_recognition.result.result.qiangzhidaoju`,
  features.`img-processor-porn.terror_recognition.result.result.zhengchangzongjiao`,
  features.`img-processor-porn.terror_recognition.result.result.youxiqiangzhidaoju`,
  features.`img-processor-porn.terror_recognition.result.result.ertongxiedian`,
  features.`img-processor-porn.terror_recognition.result.result.xuexingchangjing`,
  features.`img-processor-porn.terror_recognition.result.result.anheidongman`,
  features.`imgdetector.nmviolence.label`,
  features.`imgdetector.nmviolence.rate`,
  features.`rule-engine.description`,
  features.`data.text`,
  features.`data.origin_md5`
  """)


#save_path_shumei = "/user/zhoushengyao/diff/" + date1 + "/shumei_all"
#df_result.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_shumei)
df_result.registerTempTable("t_img_ae_shumei_all")
def save_data(data_key):
  key_list = ['shumei_kongbuzuzhi_rate','shumei_baoluanchangjing_rate','shumei_guoqiguohui_rate','shumei_junzhuang_rate','shumei_qiangzhidaoju_rate','shumei_xuexingchangjing_rate','shumei_zhengchangzongjiao_rate','shumei_ertongxiedian_rate','shumei_youxiqiangzhidaoju_rate','shumei_anheidongman_rate']
  '''
  line = ' select requestId,organization,imgUrlFile,' + str(data_key) +' ,md5 from  t_img_ae_shumei_all where '+str(data_key) + ' >= ' + str(key_list[0]) +'and'+ str(data_key) +'>=' +str(key_list[1]) \
	+'and'+ str(data_key) + '>=' +str(key_list[2]) +'and'+ str(data_key) + '>=' +str(key_list[3])+'and'++ str(data_key) + '>=' +str(key_list[4]) +'and'+ str(data_key) + '>=' +str(key_list[5]) \
	+'and'+ str(data_key) + '>=' +str(key_list[6])+'and'+ str(data_key) + '>=' +str(key_list[7])+'and'+ str(data_key) + '>=''+str(key_list[8]) +'and' + str(data_key) + '>=''+str(key_list[9])
  '''
  line = "select requestId,organization,imgUrlFile," + str(data_key) +" ,md5 from  t_img_ae_shumei_all where shumei_normal_rate!='zhengchang' and "+str(data_key) + ">=" + str(key_list[0]) +" and "+ str(data_key) + ">=" +str(key_list[1])+" and "+ str(data_key) + ">=" +str(key_list[2]) \
		         +" and "+ str(data_key) + ">=" +str(key_list[3]) \
		      +" and "+ str(data_key) + ">=" +str(key_list[4]) \
		      +" and "+ str(data_key) + ">=" +str(key_list[5]) \
		      +" and "+ str(data_key) + ">=" +str(key_list[6]) \
		      +" and "+ str(data_key) + ">=" +str(key_list[7]) \
		      +" and "+ str(data_key) + ">=" +str(key_list[8]) \
		      +" and "+ str(data_key) + ">=" +str(key_list[9])

  df_result_terror = sqlContext.sql(line)
  save_path_shumei = "diff/" + date1 + '/violence/' + data_key.replace('_rate','')
  helper.check_hdfs(save_path_shumei)

  df_result_terror.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_shumei)

key_list = ['shumei_anheidongman_rate','shumei_baoluanchangjing_rate','shumei_xuexingchangjing_rate','shumei_ertongxiedian_rate']
for key_data in key_list:
  save_data(key_data)


#df_result_terror = sqlContext.sql(""" select requestId,organization,imgUrlFile,tupu_terror_label,tupu_terror_rate, md5 from t_img_ae_shumei_all where tupu_terror_label != 0""")
#save_path_shumei = "diff/" + date1 + "/tupu_terror"
#df_result_terror.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_shumei)


df_result_terror = sqlContext.sql(""" select * from t_img_ae_shumei_all where ISNULL(shumei_baoluanchangjing_rate) = False limit 100000""")
save_path_shumei = "diff/" + date1 + "/violence/shumei_100k_terror"
helper.check_hdfs(save_path_shumei)
df_result_terror.repartition(numPartitions=1).write.format("csv").option("header", "true").option("delimiter", "\t").save(save_path_shumei)
