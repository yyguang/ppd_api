// Author Tony Tsao <cao-teng-teng.cao@foxmail.com> 2018-04-02 17:40:43
package com.shumei

import org.apache.spark.{SparkConf, SparkContext}
import com.shumei.util.{JsonSmartUtil, JacksonUtil, TimeUtil, CsvUtil, CommonUtil}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import net.minidev.json.JSONArray
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.spark.HashPartitioner

class DownloadAdData {

}

object DownloadAdData {

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
  def main(args : Array[String]): Unit = {
    val conf = new SparkConf().setAppName("com.shumei.DownloadAdData")
    val sc = new SparkContext(conf)

    //val logPath = "/user/data/event/detail_ae/dt=" + args(0) + "/serviceId=POST_IMG/"
    //val logPath = "/user/data/event/detail_ae/dt=20180212/serviceId=POST_IMG/"
    //val logPath = "/user/data/event/detail_ae/dt=2018030[1-9]/serviceId=POST_IMG/,/user/data/event/detail_ae/dt=2018031[0-9]/serviceId=POST_IMG/,/user/data/event/detail_ae/dt=2018032[0-9]/serviceId=POST_IMG/,/user/data/event/detail_ae/dt=2018033[0-1]/serviceId=POST_IMG/,/user/data/event/detail_ae/dt=20180401/serviceId=POST_IMG/"
    val logPath = "/user/data/event/detail_ae/dt=20180403/serviceId=POST_IMG/"
    val outputPath = "result/DownloadData_Ad/20180403"
    val desiredOrg = "RaI8hPdp9MjVY5OsMCqH";

    // 开始处理log，返回("url", "org")
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
          val req_type = jsonGet[String](features, "type")
          if (req_type == null || req_type.indexOf("AD") == -1)
            None
          else {
            val org = jsonGet[String](obj, "organization")
            val model = jsonGet[String](features, "rule-engine.model")
            assert (org != null)
            assert (org.indexOf(" ") == -1)
            if (org != desiredOrg || model == null)
              None
            else {
              if(model.indexOf("M011") != -1 || model.indexOf("M065") != -1 || model.indexOf("M09") != -1 || model.indexOf("M08") != -1) 
                Some(url, org)
              else
                None
            }
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
    //.sortByKey()
    .saveAsTextFile(outputPath)
  }
}
