package com.shumei.common

import com.shumei.util.JacksonUtil
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.{SparkConf, SparkContext}

class JoinData {

}

object JoinData {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("com.shumei.common.JoinData")
    val sc = new SparkContext(conf)

//    val joinPath = "/user/shiyufeng/tmp/zhangzhong_sample.json,/user/shiyufeng/tmp/format_sample_de.json"
//    val joinKey = "imei"
//    val rawPath = "/user/shiyufeng/app/for_shaowei_20171114/"
//    val rawKey = "imei"
//    val outputPath = "/user/shiyufeng/tmp/BasicFeaturesByImei_JoinData_20171114"

    val joinPath = "/user/shiyufeng/tmp/format_sample.json,/user/shiyufeng/tmp/en_sample.json"
    val joinKey = "imei"
    //val rawPath = "/user/data/temp/for_shiyufeng/fp_feature_for_shaowei_20171112"
    val rawPath = "/user/data/temp/for_shiyufeng/ip_feature_for_shaowei_20171113/"
    val rawKey = "USER_ID"
    //val outputPath = "/user/data/temp/for_shiyufeng/fp_feature_for_shaowei_20171114_join_sample"
    val outputPath = "/user/data/temp/for_shiyufeng/ip_feature_for_shaowei_20171114_join_sample"




    val joinRdd = sc.textFile(joinPath)
    val mapJoinRdd = (joinRdd
      .flatMap(line => {
        try {
          val obj = JacksonUtil.loads(line, classOf[Map[String, Any]])
          val key = obj.get(joinKey).get.toString
          Some((key, obj))
        } catch {
          case _ : Exception => None
        }
      })
      .reduceByKey((x1, x2) => {
        x1
      })
      .map(v => {
        val obj = v._2
        obj
      })
      )

    val mapJoinData = mapJoinRdd.collect()
    val mapJoinMap = scala.collection.mutable.Map[String, Map[String, Any]]()
    for (s <- mapJoinData) {
      val key = s.get(joinKey).get.toString
      mapJoinMap.put(key, s)
    }

    val bdMapSampleMap = sc.broadcast(mapJoinMap)

    val rawRdd = sc.textFile(rawPath)
    val rawMapRdd = (rawRdd
        .flatMap(line => {
          try {
            val obj = JacksonUtil.loads(line, classOf[Map[String, Any]])
            val key = obj.get(rawKey).get.toString
            if (bdMapSampleMap.value.contains(key)) {
              Some(obj ++ bdMapSampleMap.value.get(key).get)
            } else {
              None
            }
          } catch {
            case _ : Exception => None
          }
        })
    )

    (rawMapRdd
        .map(obj => JacksonUtil.dumps(obj))
        .saveAsTextFile(outputPath, classOf[GzipCodec])
    )
  }
}