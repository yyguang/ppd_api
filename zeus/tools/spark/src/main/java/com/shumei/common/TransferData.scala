package com.shumei.common

import java.io.Serializable

import com.shumei.util.{FileUtil, JacksonUtil}
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class TransferData extends Serializable {

}


object TransferData {
  /**
    * Transfer data from fromPath to toPath when successFilePath exists. The content of fromPath must json file.
    * @param sc
    * @param fromPath
    * @param successFilePath
    * @param toPath
    * @param filter Map[String, Object] => Option[Map[String, Object] ]
    * @return If success, return output rdd; else return null
    */
  def transferJsonData(sc : SparkContext, fromPath : String, successFilePath : String,
                   toPath : String, filter: Map[String, Object] => Option[Map[String, Object]]): RDD[Map[String, Object]] = {
    try {
      if (FileUtil.exist(successFilePath)) {
        val rawInputRdd = sc.textFile(fromPath).map(line => JacksonUtil.loads(line, classOf[Map[String, Object]]))
        val outputRdd = rawInputRdd.flatMap(obj => filter(obj))
        outputRdd.map(obj => JacksonUtil.dumps(obj)).saveAsTextFile(toPath, classOf[GzipCodec])
        return outputRdd
      } else {
        println(String.format("successFilePath cannot be read, path: %s", successFilePath))
        return null
      }
    } catch {
      case e : Exception => {
        println(ExceptionUtils.getFullStackTrace(e))
        return null
      }
    }
  }
}
