// Author Tony Tsao <cao-teng-teng.cao@foxmail.com> 2017-12-13 11:28:43
package com.shumei.profile.app

import org.apache.spark.{SparkConf, SparkContext}
import com.shumei.util.{JsonSmartUtil, JacksonUtil, TimeUtil, CsvUtil, CommonUtil, HBaseUtil}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import com.shumei.util.HBaseUtil.{ProfileType, Windows}
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import net.minidev.json.JSONArray
import org.apache.commons.lang.exception.ExceptionUtils

class Save360AppFeatures {

}

object Save360AppFeatures {
  // 提取信息，计算特征，输出特征
  def extractFeatures(obj: Object, window: String, backTrackingIdx: Int): Map[String, Any] = {
    val maps = JsonSmartUtil.ConvertObjToScalaObj(obj, classOf[Map[String, String]])
    val rst = mutable.Map[String, Any]()
    val keyBackTracking = window + ".backtracking_ts" 
    for (map <- maps) {
      val key = map._1
      val value = map._2
      if (key == "imei")
        rst += ("imei" -> value)
      else if (key.startsWith(window) && key != keyBackTracking) {
        var newKey = key.substring(key.indexOf(".") + 1).replace('.', '_')
        val newValue = value.split(",")(backTrackingIdx)
        if (newKey.endsWith("count")) {
          newKey = "fp_i_360_" + newKey
          rst += (newKey -> newValue.toInt)
        }
        else if (newKey.endsWith("ratio")) {
          newKey = "fp_d_360_" + newKey
          rst += (newKey -> newValue.toDouble)
        }
        else
          println("unknown item: " + key)
      }
    }
    rst.toMap
  }

  // 主函数
  def main(args : Array[String]): Unit = {
    val conf = new SparkConf().setAppName("com.shumei.profile.app.Save360AppFeatures")
    val sc = new SparkContext(conf)

    //val filePath = args(0) //"/user/data/event/detail_ae/dt=20171111/*"
    //val numBackTracking = args(1).toInt  // 处理文件中多少个回溯时间，范围为[1,6]，1的话表示只处理最近的一个回溯时间
    val filePath = "360-apps/sekeeper/result_recomm_false_*"
    val numBackTracking = 1
    val zkQuorum = "10.141.54.163,10.141.12.185,10.141.31.7,10.141.16.105,10.141.40.132"
    val zkClientPort = 2181
    
    val fileInput = sc.textFile(filePath)

    // 获取所有回溯时间
    val allBackTrackingTs = fileInput.take(1).flatMap(line => {
      var obj = JsonSmartUtil.loads(line)
      val maps = JsonSmartUtil.ConvertObjToScalaObj(obj, classOf[mutable.Map[String, String]])
      maps("all.backtracking_ts").asInstanceOf[String].split(",")
    })

    var strBackTrackings_ = ArrayBuffer[String]()
    for (i <- 0 to (numBackTracking - 1)) {
      val ts = allBackTrackingTs(i).toLong
      val date = new java.text.SimpleDateFormat("yyyyMMdd").format(new java.util.Date(ts * 1000))
      strBackTrackings_ += date
    }
    val strBackTrackings = strBackTrackings_.toArray

    // 开始处理文件
    // 分别处理n个回溯时间的时间戳，时间窗口为30d，60d和all
    val windows = Array("30d", "60d", "all")
    for (idxBackTracking <- 0 to (numBackTracking - 1)) {
      val strBackTracking = strBackTrackings(idxBackTracking)
      for (window <- windows) {
        var windowName_ = Windows._30d 
        if (window == "60d")
          windowName_ = Windows._60d
        else if (window == "all")
          windowName_ = Windows.all
        val windowName = windowName_
        val tableName = HBaseUtil.getTableName(ProfileType.imei, windowName, strBackTracking)
        val hbaseJobConf = HBaseUtil.getJobConf(tableName, zkQuorum, zkClientPort)
        val fileRdd = fileInput
        .flatMap(line => {
          try {
            var obj = JsonSmartUtil.loads(line)
            Some(extractFeatures(obj, window, idxBackTracking))
          } catch {
            case e: Exception => {
              println("##### line: " + line)
              println(ExceptionUtils.getFullStackTrace(e))
              None
            }
          }
        })
        //.collect()
        //fileRdd.foreach{println}
        .map(obj => {
          val put = HBaseUtil.mkPut(obj("imei").asInstanceOf[String], obj)
          (new ImmutableBytesWritable, put)
        })
        .saveAsNewAPIHadoopDataset(hbaseJobConf)
      }
    }
  }
}
