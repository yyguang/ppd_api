// Author Tony Tsao <cao-teng-teng.cao@foxmail.com> 2018-02-12 21:41:43
package com.shumei

import org.apache.spark.{SparkConf, SparkContext}
import com.shumei.util.{JsonSmartUtil, JacksonUtil, TimeUtil, CsvUtil, CommonUtil}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import net.minidev.json.JSONArray
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.spark.HashPartitioner

class SampleOnlineData {

}

object SampleOnlineData {

  // 获取json中某个key的value，返回值有可能为null
  def jsonGet[T](a: Object, key: String) = JsonSmartUtil.getValueFromObjectBySpecKey(a, key).asInstanceOf[T]

  // 获取json中某个key的value，如果不存在，返回默认值default
  def jsonGet[T](a: Object, key: String, default: String) = {
    val rst = JsonSmartUtil.getValueFromObjectBySpecKey(a, key).asInstanceOf[T]
    if (rst == null)
      default
    else
      rst
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
  // 第一个参数为日期
  // 第二个参数为数量
  // 第三个参数为Porn或Violence
  def main(args : Array[String]): Unit = {
    val conf = new SparkConf().setAppName("com.shumei.SampleOnlineData")
    val sc = new SparkContext(conf)

    val logPath = "/user/data/event/detail_ae/dt=" + args(0) + "/serviceId=POST_IMG/"
    val numSample = args(1).toInt
    val outputPath = "result/SampleOnlineData_" + args(2) + "/"
    val fileSample = outputPath + args(0) + "_sample"
    val mapParam = (numSample, fileSample)
    
    // 开始处理log，返回x("url", "org")
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
          var target = features
          if (args(2) == "Violence"){
            target = JsonSmartUtil.getValueFromObjectBySpecKey(features, "img-processor-porn.terror_recognition.result.result.normal") 
          } 
          else if (args(2) == "Porn"){
            target = JsonSmartUtil.getValueFromObjectBySpecKey(features, "img-processor-porn.porn_recognition.result.porn")
          }
          else {target = null}
          if (target == null)
            None
          else {
            val org = jsonGet[String](obj, "organization")
            assert (org != null)
            assert (org.indexOf(" ") == -1)
            if (org == "osqmahrbZ4R4s7TXikbI" || org == "fe1iHzsMKi3vHLmqtHyl")  // 去掉华数/数美运维账号的数据
              None
            else
              Some(url, org)
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
    .partitionBy(new HashPartitioner(10))
    .persist()

    // get counts
    val param = mapParam
    val rate = clip(param._1 / logRdd.count().toDouble, 0.0, 1.0)
    logRdd.sample(false, rate)
    .repartition(1)
    .saveAsTextFile(param._2)
  }
}
