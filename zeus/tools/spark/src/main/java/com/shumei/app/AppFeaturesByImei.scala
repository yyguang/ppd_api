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

class AppFeaturesByImei {

}

object AppFeaturesByImei {

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
  
  // 提取hookApps等
  def jsonGetKeySet(obj: Object, key: String): Set[String] = {
    val map = JsonSmartUtil.getValueFromObjectBySpecKey(obj, key)
    if (map == null) Set[String]() else JsonSmartUtil.ConvertObjToScalaObj(map, classOf[Map[String, String]]).keySet
  }
  
  // 提取android系统的app相关信息，计算特征
  def extractAndroid(obj: Object, paydayAppsSet: Set[String], nonPaydayAppsSet: Set[String]): Map[String, Any] = {
    val data = JsonSmartUtil.getValueFromObjectBySpecKey(obj, "data")
    val smid = jsonGet[String](data, "smid", "")
    val imei = jsonGet[String](data, "imei", "")

    val hookApps = jsonGetKeySet(obj, "hookApps")
    val monkeyApps = jsonGetKeySet(obj, "monkeyApps")
    val vpnApps = jsonGetKeySet(obj, "vpnApps")
    val alterApps = jsonGetKeySet(obj, "alterApps")
    val alterLocApps = jsonGetKeySet(obj, "alterLocApps")
    val alterApkApps = jsonGetKeySet(obj, "alterApkApps")
    val remoteCtrApps = jsonGetKeySet(obj, "remoteCtrApps")
    val multiBoxingApps = jsonGetKeySet(obj, "multiBoxingApps")

    val paydayApps = mutable.Set[String]()
    val nonPaydayApps = mutable.Set[String]()
    var paydayLastTs = 0L
    var paydayLastActiveTs = 0L
    var nonPaydayLastTs = 0L
    var nonPaydayLastActiveTs = 0L
    
    val activeApp = jsonGet[String](data, "appname")
    val activeTs = TimeUtil.timeToTimestamp(jsonGet[Number](data, "t").longValue)
    val appver = jsonGet[String](data, "appver")
    val activeAppList = Map[String, Tuple2[Long, String]](activeApp -> (activeTs, appver))
    if (paydayAppsSet.contains(activeApp))
      paydayLastActiveTs = math.max(paydayLastActiveTs, activeTs)
    else if (nonPaydayAppsSet.contains(activeApp))
      nonPaydayLastActiveTs = math.max(nonPaydayLastActiveTs, activeTs)
    
    val apps_ = JsonSmartUtil.getValueFromObjectBySpecKey(data, "apps")
    val apps = if (apps_ == null) List[String]() else JsonSmartUtil.ConvertObjToScalaObj(apps_, classOf[List[String]])
    var intSecAppsCount_ = -1
    var totalAppsCount_ = -1
    if (apps_ != null) {
      intSecAppsCount_ = jsonGet[String](obj, "intSecAppsCount").toInt
      totalAppsCount_ = jsonGet[String](obj, "totalAppsCount").toInt
    }
    val intSecAppsCount = intSecAppsCount_
    val totalAppsCount = totalAppsCount_

    val appList = mutable.Map[String, Tuple2[Long, Int]]()
    for (app <- apps) {
      var words: Array[String] = null
      try {
        words = app.split(",")
      }
      catch {
        case e: Exception => {
          println("##### app: " + app)
          println(ExceptionUtils.getFullStackTrace(e))
        }
      }
      if (words != null && words.size > 1) {
        val ts = TimeUtil.timeToTimestamp(words(0).toLong)
        val appName = words(1)
        if (paydayAppsSet.contains(appName)) {
          paydayApps += appName
          paydayLastTs = math.max(paydayLastTs, ts)
        }
        else if (nonPaydayAppsSet.contains(appName)) {
          nonPaydayApps += appName
          nonPaydayLastTs = math.max(nonPaydayLastTs, ts)
        }
        val user = if (words.size == 4) words(3).toInt else -1
        appList += (appName -> (ts, user))
      }
    }

    Map[String, Any](
      "os" -> "android",
      "smid" -> smid,
      "imei" -> imei,
      "hookAppList" -> hookApps,
      "monkeyAppList" -> monkeyApps,
      "vpnAppList" -> vpnApps,
      "alterAppList" -> alterApps,
      "alterLocAppList" -> alterLocApps,
      "alterApkAppList" -> alterApkApps,
      "remoteCtrAppList" -> remoteCtrApps,
      "multiBoxingAppList" -> multiBoxingApps,
      "paydayAppList" -> paydayApps.toSet,
      "paydayLastTs" -> paydayLastTs,
      "paydayLastActiveTs" -> paydayLastActiveTs,
      "nonPaydayAppList" -> nonPaydayApps.toSet,
      "nonPaydayLastTs" -> nonPaydayLastTs,
      "nonPaydayLastActiveTs" -> nonPaydayLastActiveTs,
      "appList" -> appList.toMap,
      "activeAppList" -> activeAppList,
      "intSecAppsCount" -> (activeTs, intSecAppsCount, totalAppsCount)
    )
  }

  // 提取ios系统的app相关信息，计算特征
  def extractIos(obj: Object): Map[String, Any] = {
    val data = JsonSmartUtil.getValueFromObjectBySpecKey(obj, "data")
    val smid = jsonGet[String](data, "smid", "")
    val idfa = jsonGet[String](data, "idfa", "")
    val hookApps = jsonGetKeySet(obj, "hookApps")
    val monkeyApps = jsonGetKeySet(obj, "monkeyApps")
    val vpnApps = jsonGetKeySet(obj, "vpnApps")
    
    Map[String, Any](
      "os" -> "ios",
      "smid" -> smid,
      "idfa" -> idfa,
      "hookAppList" -> hookApps,
      "monkeyAppList" -> monkeyApps,
      "vpnAppList" -> vpnApps
    )
  }

  // 生成一个新的set，为x(key)和y(key)的并集，其中这两个集合类型为Set[String]
  def unionSet(x: Map[String, Any], y: Map[String, Any], key: String): Set[String] = {
    (x(key).asInstanceOf[Set[String]] ++ y(key).asInstanceOf[Set[String]])
  }

  // 合并特征
  def mergeFeatures(x: Map[String, Any], y: Map[String, Any]): Map[String, Any] = {
    if (x("os") == "ios")
      Map[String, Any](
        "os" -> "ios",
        "hookAppList" -> unionSet(x, y, "hookAppList"),
        "monkeyAppList" -> unionSet(x, y, "monkeyAppList"),
        "vpnAppList" -> unionSet(x, y, "vpnAppList")
      )
    else {
      val xIntSecAppsCount = x("intSecAppsCount").asInstanceOf[Tuple3[Long, Int, Int]]
      val yIntSecAppsCount = y("intSecAppsCount").asInstanceOf[Tuple3[Long, Int, Int]]
      
      Map[String, Any](
        "os" -> "android",
        "hookAppList" -> unionSet(x, y, "hookAppList"),
        "monkeyAppList" -> unionSet(x, y, "monkeyAppList"),
        "vpnAppList" -> unionSet(x, y, "vpnAppList"),
        "alterAppList" -> unionSet(x, y, "alterAppList"),
        "alterLocAppList" -> unionSet(x, y, "alterLocAppList"),
        "alterApkAppList" -> unionSet(x, y, "alterApkAppList"),
        "remoteCtrAppList" -> unionSet(x, y, "remoteCtrAppList"),
        "multiBoxingAppList" -> unionSet(x, y, "multiBoxingAppList"),
        "paydayAppList" -> unionSet(x, y, "paydayAppList"),
        "paydayLastTs" -> math.max(x("paydayLastTs").asInstanceOf[Long], y("paydayLastTs").asInstanceOf[Long]),
        "paydayLastActiveTs" -> math.max(x("paydayLastActiveTs").asInstanceOf[Long], y("paydayLastActiveTs").asInstanceOf[Long]),
        "nonPaydayAppList" -> unionSet(x, y, "nonPaydayAppList"),
        "nonPaydayLastTs" -> math.max(x("nonPaydayLastTs").asInstanceOf[Long], y("nonPaydayLastTs").asInstanceOf[Long]),
        "nonPaydayLastActiveTs" -> math.max(x("nonPaydayLastActiveTs").asInstanceOf[Long], y("nonPaydayLastActiveTs").asInstanceOf[Long]),
        "appList" -> (x("appList").asInstanceOf[Map[String, Tuple2[Long, Int]]] ++ y("appList").asInstanceOf[Map[String, Tuple2[Long, Int]]]),
        "activeAppList" -> (x("activeAppList").asInstanceOf[Map[String, Tuple2[Long, String]]] ++ y("activeAppList").asInstanceOf[Map[String, Tuple2[Long, String]]]),
        "intSecAppsCount" -> (if (xIntSecAppsCount._1 > yIntSecAppsCount._1) xIntSecAppsCount else yIntSecAppsCount)
      )
    }
  }

  // 从map中获取某个key并转化为相应类型
  def mapGet[T](x: Map[String, Any], key: String) = x(key).asInstanceOf[T]

  // 输出特征
  def outputFeatures(x: Map[String, Any]): Map[String, Any] = {
    val hookApps = mapGet[Set[String]](x, "hookAppList")
    val monkeyApps = mapGet[Set[String]](x, "monkeyAppList")
    val vpnApps = mapGet[Set[String]](x, "vpnAppList")
    val rst = mutable.Map[String, Any](
      "fp_s_hook_app_set" -> JacksonUtil.dumps(hookApps),
      "fp_i_hook_app_setcount" -> hookApps.size,
      "fp_s_monkey_app_set" -> JacksonUtil.dumps(monkeyApps),
      "fp_i_monkey_app_setcount" -> monkeyApps.size,
      "fp_s_vpn_app_set" -> JacksonUtil.dumps(vpnApps),
      "fp_i_vpn_app_setcount" -> vpnApps.size
    )
    if (x("os") == "android") {
      val alterApps = mapGet[Set[String]](x, "alterAppList")
      val alterLocApps = mapGet[Set[String]](x, "alterLocAppList")
      val alterApkApps = mapGet[Set[String]](x, "alterApkAppList")
      val remoteCtrApps = mapGet[Set[String]](x, "remoteCtrAppList")
      val multiBoxingApps = mapGet[Set[String]](x, "multiBoxingAppList")
      val paydayApps = mapGet[Set[String]](x, "paydayAppList")
      val nonPaydayApps = mapGet[Set[String]](x, "nonPaydayAppList")
      val appList = x("appList").asInstanceOf[Map[String, Tuple2[Long, Int]]]   
      val activeAppList = x("activeAppList").asInstanceOf[Map[String, Tuple2[Long, String]]]
      rst ++= Map[String, Any](
        "fp_s_alter_app_set" -> JacksonUtil.dumps(alterApps),
        "fp_i_alter_app_setcount" -> alterApps.size,
        "fp_s_alter_loc_app_set" -> JacksonUtil.dumps(alterLocApps),
        "fp_i_alter_loc_app_setcount" -> alterLocApps.size,
        "fp_s_alter_apk_app_set" -> JacksonUtil.dumps(alterApkApps),
        "fp_i_alter_apk_app_setcount" -> alterApkApps.size,
        "fp_s_remote_ctr_app_set" -> JacksonUtil.dumps(remoteCtrApps),
        "fp_i_remote_ctr_app_setcount" -> remoteCtrApps.size,
        "fp_s_multi_boxing_app_set" -> JacksonUtil.dumps(multiBoxingApps),
        "fp_i_multi_boxing_app_setcount" -> multiBoxingApps.size,
        "fp_s_payday_app_set" -> JacksonUtil.dumps(paydayApps),
        "fp_i_payday_app_setcount" -> paydayApps.size,
        "fp_l_payday_last_ts_max" -> x("paydayLastTs").asInstanceOf[Long],
        "fp_l_payday_last_active_ts_max" -> x("paydayLastActiveTs").asInstanceOf[Long],
        "fp_s_loan_app_set" -> JacksonUtil.dumps(nonPaydayApps),
        "fp_i_loan_app_setcount" -> nonPaydayApps.size,
        "fp_l_loan_last_ts_max" -> x("nonPaydayLastTs").asInstanceOf[Long],
        "fp_l_loan_last_active_ts_max" -> x("nonPaydayLastActiveTs").asInstanceOf[Long],
        "fp_s_app_list_other" -> JacksonUtil.dumps(appList),
        "fp_s_active_app_list_other" -> JacksonUtil.dumps(activeAppList),
        "fp_i_app_cnt_other" -> appList.size
      )

      // userAppCnt特征处理
      var userApps = 0
      var unknownApps = 0
      val appListValues = appList.values
      for (app <- appListValues) {
        if (app._2 == 1)
          userApps += 1
        else if (app._2 == -1)
          unknownApps += 1
      }
      if (unknownApps != appList.size)
        rst += ("fp_i_user_app_cnt_other" -> userApps)
      
      // 整秒安装app相关特征处理
      val intTotalSecAppsCount = x("intSecAppsCount").asInstanceOf[Tuple3[Long, Int, Int]]
      val intSecAppsCount = intTotalSecAppsCount._2
      val totalAppsCount = intTotalSecAppsCount._3
      var unintSecAppsCount = -1
      var appIntSecRate = -1.0
      if (totalAppsCount != -1.0 && intSecAppsCount != -1.0) {
        unintSecAppsCount = totalAppsCount - intSecAppsCount
        if (totalAppsCount != 0)
          appIntSecRate = intSecAppsCount / totalAppsCount.toDouble
      }
      if (appIntSecRate != -1.0)
        rst += ("fp_d_collect_app_int_sec_rate_latest" -> appIntSecRate)
      if (intSecAppsCount != -1)
        rst += ("fp_i_collect_app_cnt_int_sec_latest" -> intSecAppsCount)
      if (unintSecAppsCount != -1)
        rst += ("fp_i_collect_app_cnt_unint_sec_latest" -> unintSecAppsCount)
    }
    // 过滤所有类型为Set[String]且为空的
    rst.filter((t) => {
      try {
        if (t._2.asInstanceOf[Set[String]].isEmpty) false else true
      }
      catch {
        case e: Exception => true
      }
    })
    rst.toMap
  }

  // 主函数
  def main(args : Array[String]): Unit = {
    val conf = new SparkConf().setAppName("com.shumei.profile.app.AppFeaturesByImei")
    val sc = new SparkContext(conf)

//    val fpPath = args(0)
//    val app2CatPath = args(1)
//    val outputPath = args(2)
//    val outputPathNumPartitions = args(3).toInt

    //val fpPath = "head10.txt"
    val fpPath = args(0) //"/user/data/event/detail_ae/dt=20171111/*"
    val strBackTracking = args(1) //"20171111"
    val app2CatPath = "package2Category-20171208.csv"
    //val outputPath = "result/AppFts"
    //val outputPartitions = 10
    val zkQuorum = "10.141.54.163,10.141.12.185,10.141.31.7,10.141.16.105,10.141.40.132"
    val zkClientPort = 2181
    
    // 获取app2Cat列表
    val rddApp2Cat = CsvUtil.loadCsv(sc, app2CatPath)
    rddApp2Cat.persist()
  
    // 构建两个Set：paydayApps和nonPaydayApps
    val paydayAppsBuf = mutable.Set[String]()
    val nonPaydayAppsBuf = mutable.Set[String]()
    val paydayAll = rddApp2Cat.filter(line => line(3).trim() == "payday")
    .map(line => line(0).trim())
    .collect()
    for (item <- paydayAll)
      paydayAppsBuf += item
    val nonPaydayAll = rddApp2Cat.filter(line => {
      val cat = line(3).trim()
      cat != "payday" && cat != "non_loan"
    })
    .map(line => line(0).trim())
    .collect()
    for (item <- nonPaydayAll)
      nonPaydayAppsBuf += item
    val paydayAppsSet = paydayAppsBuf.toSet
    val nonPaydayAppsSet = nonPaydayAppsBuf.toSet

    // 开始处理fp
    val fpInput = sc.textFile(fpPath)
    val fpRdd = fpInput
    .flatMap(line => {
      try {
        var obj = JsonSmartUtil.loads(line)
        val requestType = JsonSmartUtil.getValueFromObjectBySpecKey(obj, "requestType").asInstanceOf[String]
        if (requestType != "all")
          None
        else {
          val os = JsonSmartUtil.getValueFromObjectBySpecKey(obj, "data@@@os").asInstanceOf[String]
          if (os == "android")
            Some(extractAndroid(obj, paydayAppsSet, nonPaydayAppsSet))
          else if (os == "ios")
            Some(extractIos(obj))
          else
            None
        }
      } catch {
        case e: Exception => {
          println("##### line: " + line)
          println(ExceptionUtils.getFullStackTrace(e))
          None
        }
      }
    })
    //fpRdd.count()
    fpRdd.persist()

    // 统计smid
    val windows = Windows._1d
    val tableName = HBaseUtil.getTableName(ProfileType.smid, windows, strBackTracking)
    val hbaseJobConf = HBaseUtil.getJobConf(tableName, zkQuorum, zkClientPort)
    
    //println(fpRdd.filter(x => (x("smid").asInstanceOf[String] == null)).count().toString)
    val resultRdd = fpRdd.filter(x => {x("smid").asInstanceOf[String] != ""})
    .map(x => (x("smid").asInstanceOf[String], x))
    .reduceByKey((x,y) => {
      try {
        mergeFeatures(x,y)
      } catch {
        case e: Exception => {
          println("##### value1: " + x)
          println("##### value2: " + y)
          println(ExceptionUtils.getFullStackTrace(e))
          Map[String, Any]()
        }
      }
    })
    .filter{case(x,y) => !y.isEmpty}
    .mapValues(x => outputFeatures(x))
     
    //resultRdd.count()
    .map(obj => {
      val put = HBaseUtil.mkPut(obj._1.toString, obj._2)
      (new ImmutableBytesWritable, put)
    })
    .saveAsNewAPIHadoopDataset(hbaseJobConf)
    
    // 统计imei

    // 统计idfa
  }
}
