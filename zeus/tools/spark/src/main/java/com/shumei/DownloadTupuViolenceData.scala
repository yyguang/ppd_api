// Author Tony Tsao <cao-teng-teng.cao@foxmail.com> 2018-03-14 11:52:43
package com.shumei

import org.apache.spark.{SparkConf, SparkContext}
import com.shumei.util.{JsonSmartUtil, JacksonUtil, TimeUtil, CsvUtil, CommonUtil}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import net.minidev.json.JSONArray
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.spark.HashPartitioner

class DownloadTupuViolenceData {

}

object DownloadTupuViolenceData {

  // 获取json中某个key的value，返回值有可能为null
  // 有可能会抛出异常
  def jsonGet[T](a: Object, key: String) = JsonSmartUtil.getValueFromObjectBySpecKey(a, key).asInstanceOf[T]

  // 获取json中某个key的value，如果不存在，返回默认值default
  def jsonGet[T](a: Object, key: String, default: T) = {
    val rst = JsonSmartUtil.getValueFromObjectBySpecKey(a, key)
    if (rst != null)
      rst.asInstanceOf[T] // 仍然有可能抛出异常
    else
      default
  }
  
  // 限制x的范围在minValue和maxValue之间
  def clip(x: Double, minValue: Double, maxValue: Double): Double = {
    if (x > maxValue)
      maxValue
    else if (x < minValue)
      minValue
    else
      x
  }

  // 主函数
  def main(args : Array[String]): Unit = {
    val conf = new SparkConf().setAppName("com.shumei.DownloadTupuViolenceData")
    val sc = new SparkContext(conf)

    //val logPath = "/user/data/event/detail_ae/dt=" + args(0) + "/serviceId=POST_IMG/"
    //val logPath = "/user/data/event/detail_ae/dt=20180212/serviceId=POST_IMG/"
    val logPath = "/user/data/event/detail_ae/dt=2018021[2-9]/serviceId=POST_IMG/,/user/data/event/detail_ae/dt=2018022[0-6]/serviceId=POST_IMG/,/user/data/event/detail_ae/dt=2018030[0-9]/serviceId=POST_IMG/,/user/data/event/detail_ae/dt=2018031[0-2]/serviceId=POST_IMG/"
    val outputPath = "result/DownloadData_TupuViolence/20180212-20180312"

    //#imgdetector.nmviolence.label
    //#imgdetector.nmviolence.rate
    // 开始处理log，返回("rate", "url")
    val logInput = sc.textFile(logPath)
    val logRdd = logInput
    .flatMap(line => {
      try {
        var obj = JsonSmartUtil.loads(line)
        val features = JsonSmartUtil.getValueFromObjectBySpecKey(obj, "features")
        var url_ = jsonGet[String](features, "data.imgUrlFile")
        if (url_ == null || url_ == "")
          url_ = jsonGet[String](features, "data.imgUrl")
        if (url_ == null || url_ == "")
          url_ = jsonGet[String](obj, "data@@@imgUrlFile")
        if (url_ == null || url_ == "")
          url_ = jsonGet[String](obj, "data@@@imgUrl")
        val url = url_
        if (url == null || url == "")
          None
        else {
          val tupuLabel = jsonGet[Int](features, "imgdetector.nmviolence.label")
          if (tupuLabel == null || tupuLabel == 0)
            None
          else {
            val rate = jsonGet[Double](features, "imgdetector.nmviolence.rate")
            Some(rate, url)
          }
        }
      } catch {
        case e: Exception => {
          println("##### line: " + line)
          println(ExceptionUtils.getFullStackTrace(e))
          None
        }
      }
    })
    .repartition(1)
    .sortByKey()
    .saveAsTextFile(outputPath)
  }
}
