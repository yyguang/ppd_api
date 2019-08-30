package com.shumei.common

import com.shumei.util.JacksonUtil
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

class FlatJson {

}

object FlatJson {
  def main(args : Array[String]): Unit = {
    val conf = new SparkConf().setAppName("com.shumei.common.FlatJson")
    val sc = new SparkContext(conf)

    val rawPath = "/user/data/event/detail_ae/dt=20171111/serviceId=POST_TEXT/POST_TEXT#part-02045.gz"

    val keySetOutputPath = ""

    var flatJsonOutputPath = ""
    if (args.size >= 3) {
      val flatJsonOutputPath = args(2)
    }


    val rawRdd = sc.textFile(rawPath)

    val mapRdd = (rawRdd
        .map(line => {
          val obj = JacksonUtil.loads(line, classOf[Map[String, Any]])
          val resultMap = mutable.Map[String, Any]()
          processMap("", obj, resultMap)
          resultMap
        })
    )

    if (!flatJsonOutputPath.equals("")) {
      mapRdd.persist()

      (mapRdd
          .map(obj => JacksonUtil.dumps(obj))
          .saveAsTextFile(flatJsonOutputPath, classOf[GzipCodec])
      )
    }

    val reduceRdd = (mapRdd
        .map(obj => {
          obj.keySet
        })
        .reduce((x1, x2) => {
          x1.++(x2)
        })
    )



  }

  def processMap(keyPrefix : String, obj : Map[String, Any], resultMap : mutable.Map[String, Any]): Unit = {
    for ((k, v) <- obj) {
      var nextKeyPrefix = k
      if (!keyPrefix.equals("")) {
        nextKeyPrefix = "%s@@@%s".format(keyPrefix, k)
      }
      if (v.isInstanceOf[Map[Any, Any]]) {
        processMap(nextKeyPrefix, v.asInstanceOf[Map[String, Any]], resultMap)
      } else if (v.isInstanceOf[List[Any]]) {

      } else {
        resultMap.put(nextKeyPrefix, v)
      }
    }
  }
}