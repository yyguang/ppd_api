package com.shumei.util

import org.apache.hadoop.hbase.{HBaseConfiguration, TableName}
import org.apache.hadoop.hbase.client.{ConnectionFactory, Get, Put, Result}
import org.apache.hadoop.hbase.mapreduce.{TableInputFormat, TableOutputFormat}
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.{Accumulator, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.hadoop.hbase.io.ImmutableBytesWritable

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import com.shumei.util.JacksonUtil

class HBaseUtil extends Serializable {

}

object HBaseUtil extends Serializable {

  class ProfileType {

  }

  object ProfileType extends Enumeration {
    val smid, imei, idfa, prcid, phone, ip, token, stats, smidbrother = Value
  }

  class Windows {

  }

  object Windows extends Enumeration {
    val _1d = Value("1d")
    val _7d = Value("7d")
    val _30d = Value("30d")
    val _60d = Value("60d")
    val _90d = Value("90d")
    val all = Value("all")
  }

  class Source {

  }

  object Source extends Enumeration {
    val basic = Value("basic")
    val txt = Value("txt")
    val img = Value("img")
    val fp = Value("fp")
    val fr = Value("fr")
    val dtt = Value("dtt")
    //val lr = Value("lr")
    val lg = Value("lg")
    val rg = Value("rg")
    val global = Value("global")
  }

  class DataType {

  }

  object DataType extends Enumeration {
    val int = Value("i")
    val long = Value("l")
    val float = Value("f")
    val double = Value("d")
    val string = Value("s")
  }

  class CalType {

  }

  object CalType extends Enumeration {
    val mean = Value("mean")// sum/count
    val max = Value("max")
    val min = Value("min")
    val count = Value("count")
    val sum = Value("sum")
    val medium = Value("medium")
    val stddev = Value("stddev")
    val variance = Value("variance")
    val other = Value("other")
    val relation = Value("relation")
    val list = Value("list")
    val set = Value("set")
    val setCount = Value("setcount")
    val map = Value("map")
    val ratio = Value("ratio")
    val latest = Value("latest")
  }

  def genColName(source : Source.Value, dataType: DataType.Value, featureName : String, calType: CalType.Value): String = {
    return "%s_%s_%s_%s".format(source.toString, dataType.toString, featureName, calType.toString)
  }

  def getTableName(profileType: ProfileType.Value, windows: Windows.Value, backTrackingTs: Long): String = {
    val strBackTracking = TimeUtil.timestampToStr(backTrackingTs, "yyyyMMdd")
    return getTableName(profileType, windows, strBackTracking)
  }

  /**
    *
    * @param profileType
    * @param windows
    * @param backTracking yyyyMMdd, like: 20171101
    * @return
    */
  def getTableName(profileType: ProfileType.Value, windows: Windows.Value, backTracking: String): String = {
    return "profile_%s_%s_%s".format(profileType.toString, windows.toString, backTracking)
  }

  def getJobConf(tableName : String, zkQuorum : String = "", zkClientPort : Int = 0) : JobConf = {
    val conf = HBaseConfiguration.create()
    if (zkQuorum != "") {
      conf.set("hbase.zookeeper.quorum", zkQuorum)
    }
    if (zkClientPort != 0) {
      conf.set("hbase.zookeeper.property.clientPort", zkClientPort.toString)
    }
    conf.set("hbase.client.keyvalue.maxsize","104857600")

    val jobConf = new JobConf(conf)
    jobConf.set(TableInputFormat.INPUT_TABLE, tableName)
    jobConf.set(TableOutputFormat.OUTPUT_TABLE, tableName)
    jobConf.setOutputKeyClass(classOf[ImmutableBytesWritable])
    jobConf.setOutputValueClass(classOf[Result])
    jobConf.set("mapreduce.outputformat.class",classOf[TableOutputFormat[ImmutableBytesWritable]].getName)

    return jobConf
  }

  /**
    *
    * @param rowKey
    * @param columnMap columnMap的key按照如下命名：columnFamily_dataType_columnName，如txt_f_ad_score_mean，表示对应的value需要存放到txt的columnFamily，
    *                  并且按照float类型存储，columnName为f_ad_score.mean
    * @return
    */
  def mkPut(rowKey: String, columnMap: Map[String, Any]): Put = {
    val md5RowKey = CommonUtil.md5(rowKey)
    val put = new Put(Bytes.toBytes(md5RowKey))
    //把没有md5的数据加入到column中
    put.addColumn(Bytes.toBytes("basic"), Bytes.toBytes("no_md5_key"), Bytes.toBytes(rowKey))
    for ((key, value) <- columnMap) {
      //column family和column 之间使用_分割，如果没有_，表示不用存储到hbase中
      val words = key.split("_")
      if (words.size >= 3) {
        val columnFamily = words(0)
        val dataType = words(1)
        val columnName = words.toList.slice(2, words.size).mkString("_")
        var realValue = Array[Byte]()
        if (dataType.equals("i")) {
          realValue = Bytes.toBytes(value.asInstanceOf[Int])
        } else if (dataType.equals("f")) {
          realValue = Bytes.toBytes(value.asInstanceOf[Float])
        } else if (dataType.equals("s")) {
          if(value.isInstanceOf[String]) {
            val strValue = value.asInstanceOf[String]
            if (!strValue.equals("")) {
                realValue = Bytes.toBytes(strValue)
            }
          }else {
            realValue = Bytes.toBytes(JacksonUtil.dumps(value))
          }
        } else if (dataType.equals("d")) {
          realValue = Bytes.toBytes(value.asInstanceOf[Double])
        } else if (dataType.equals("l")) {
          realValue = Bytes.toBytes(value.asInstanceOf[Long])
        }
        put.addColumn(Bytes.toBytes(columnFamily), Bytes.toBytes("%s_%s".format(dataType, columnName)), realValue)
      }
    }
    return put
  }

//  /**
//    * 增加计数统计
//    * @param rowKey
//    * @param columnMap
//    * @param acc
//    * @return
//    */
//  def mkPut(rowKey : String, columnMap : Map[String, Any], acc : Accumulator[mutable.Map[String, Long]], appName : String) : Put = {
//    var columnFamily = ""
//    for ((k, v) <- columnMap) {
//      val words = k.split("_")
//      if (words.size >= 3) {
//        columnFamily = words(0)
//        //把所有类型转换为s，放入到acc中，acc的value是Long，但是在saveStatsToHbase的时候会被转换为string
//        val dataType = "s"
//        val columnName = words.toList.slice(2, words.size).mkString("_")
//        acc.add(mutable.Map[String, Long](
//          "%s_%s_%s_%s".format(columnFamily, dataType, columnName, CalType.count.toString) -> 1l
//        ))
//      }
//    }
//    if (columnFamily != "") {
//      //把rowKey也放入到acc中
//      acc.add(mutable.Map[String, Long](
//        "%s_%s_%s_%s".format(columnFamily, "s", appName, CalType.count.toString) -> 1l
//      ))
//    }
//    return HBaseUtil.mkPut(rowKey, columnMap)
//  }
//
//  /**
//    * 把统计值保存到hbase中
//    * @param appName
//    * @param zkQuorum
//    * @param zkClientPort
//    * @param acc
//    * @param sc
//    */
//  def saveStatsToHbase(appName : String, zkQuorum : String, zkClientPort : Int, acc : Accumulator[mutable.Map[String, Long]], sc : SparkContext,
//                       windows : Windows.Value, strBacktrackingDay : String): Unit = {
//    //把所有的long类型的value转换为string
//    val resultMap = mutable.Map[String, String]()
//    for ((k, v) <- acc.value) {
//      resultMap.put(k, v.toString)
//    }
//
//    val rowKey = "%s_%s_%s".format(appName, windows.toString, strBacktrackingDay)
//
//    val statsRdd = sc.makeRDD(List(resultMap.toMap))
//    val hbaseStatsRdd = (statsRdd
//      .map(obj => {
//        val put = new Put(Bytes.toBytes(rowKey))
//        for ((key, value) <- obj) {
//          put.addColumn(Bytes.toBytes(Source.basic.toString), Bytes.toBytes(key), Bytes.toBytes(value))
//        }
//        (new ImmutableBytesWritable, put)
//      })
//      )
//    val statsHbaseJobConf = HBaseUtil.getJobConf("profile_stats", zkQuorum, zkClientPort)
//    hbaseStatsRdd.saveAsNewAPIHadoopDataset(statsHbaseJobConf)
//
//  }


  /**
    *
    * @param sc
    * @param zkQuorum
    * @param zkClientPort
    * @param tableName
    * @param columnList 待查询的column列表，column命名：columnFamily_dataType_columnName，如global_s_ip，
    *                   表示要取global列族下s_ip列中的value
    * @return 返回元组组成的RDD，其中元组形式：(rowkey, Map((columnFamily_dataType_columnName -> value)...))，
    *         如：(BsHKz7OMBibZYAoy3ZCA_LCMluUaJm6Lqed6j1YeDXA==,Map((global_s_ip -> {"117.136.66.170":{"from":"fp","ts":1511881269000}}),
    *         (global_s_smid -> {"20170326214524a2ecb4de1bb914d431cbe74e92ef392c2472a3521032b1c7":{"from":"fp","ts":1511881269000}})))
    */
  def mkGet(sc:SparkContext, zkQuorum:String, zkClientPort:Int, tableName:String, columnList:List[String])
  :RDD[Tuple2[String,Map[String, Any]]] = {
    val hbaseconf = getJobConf(tableName, zkQuorum, zkClientPort)
    val hbaseRDD = sc.newAPIHadoopRDD(hbaseconf, classOf[TableInputFormat],
      classOf[org.apache.hadoop.hbase.io.ImmutableBytesWritable],
      classOf[org.apache.hadoop.hbase.client.Result])
    hbaseRDD.map({case(_,result) =>
      val colValueMap = mutable.Map[String,Any]()
      val rowKey = Bytes.toString(result.getValue("basic".getBytes, "no_md5_key".getBytes))
      for(col<-columnList){
        val cols = col.toString.split("_")
        if(cols.size>=3){
          try{
            val columnFamily = cols(0)
            val dataType = cols(1)
            val columnName = cols.toList.slice(2, cols.size).mkString("_")
            val value = result.getValue(columnFamily.getBytes, (dataType+"-"+columnName).getBytes)
            if (value!=null) {
              colValueMap.put(col, Bytes.toString(value))
            }
          }catch {
            case _:Exception => {}
          }
        }
      }
      (rowKey,colValueMap.toMap)
    })
  }

  /**
    *
    * @param zkQuorum
    * @param zkClientPort
    * @param tableName
    * @param RowKeyList 待查询的RowKey列表，md5加密
    * @param columnList 待查询的column列表，column命名：columnFamily_dataType_columnName，如global_s_ip，
    *                   表示要取global列族下s_ip列中的value
    * @return 返回Map，形式：Map(rowkey, Map((columnFamily_dataType_columnName -> value)...))，
    *         如：Map(105fb7449a56df182690300ac0b0d76 ->
    *         Map(global_s_smid -> {"20170326214524a2ecb4de1bb914d431cbe74e92ef392c2472a3521032b1c7":{"from":"fp","ts":1511881269000}},
    *         global_s_ip -> {"117.136.66.170":{"from":"fp","ts":1511881269000}}),
    *         10771fff46a9c8c5f8db0c89525f20 ->
    *         Map(global_s_smid -> {"20171010203118d6928890367709c81c511d7bfbd7dda3d359cbaf6176fc81":{"from":"fp","ts":1511864473000}},
    *         global_s_ip -> {"122.137.133.27":{"from":"fp","ts":1511864473000}}))
    */
  def getValueByRowKey(zkQuorum:String, zkClientPort:Int, tableName:String, RowKeyList:List[String], columnList:List[String]):
  Map[String,Any] = {
    val conf = HBaseConfiguration.create()
    conf.set("hbase.zookeeper.quorum", zkQuorum)
    conf.set("hbase.zookeeper.property.clientPort", zkClientPort.toString)
    conf.set(TableInputFormat.INPUT_TABLE, tableName)
    val conn = ConnectionFactory.createConnection(conf)
    val userTable = TableName.valueOf(tableName)
    val table = conn.getTable(userTable)
    val rowKeyToValue = mutable.Map[String,Any]()
    for (rk<-RowKeyList){
      try{
        val rkGet = new Get(Bytes.toBytes(rk))
        val HBaseRow = table.get(rkGet)
        var result:AnyRef = null
        val columnToValue = mutable.Map[String,Any]()
        if(HBaseRow != null && !HBaseRow.isEmpty){
          for(col<-columnList){
            val cols = col.toString.split("-")
            if(cols.size>=3) {
              val columnFamily = cols(0)
              val dataType = cols(1)
              val columnName = cols.toList.slice(2, cols.size).mkString("_")
              val value = Bytes.toString(HBaseRow.getValue(Bytes.toBytes(columnFamily),Bytes.toBytes(dataType+"-"+columnName)))
              columnToValue.put(col,value)
            }
          }
        }
        rowKeyToValue.put(rk,columnToValue.toMap)
      } catch {
        case _:Exception=>{}
      }
    }
    rowKeyToValue.toMap
  }

  def saveStatsToHbase(zkQuorum : String, zkClientPort : Int, acc : Accumulator[mutable.Map[String, Long]], sc : SparkContext): Unit = {
    //把所有的long类型的value转换为string
    val resultArray = ArrayBuffer[Tuple2[String, String]]()
    for ((k, v) <- acc.value) {
      resultArray.append((k, v.toString))
    }

    val statsRdd = sc.makeRDD(resultArray.toList)
    val hbaseStatsRdd = (statsRdd
      .map(obj => {
        val put = new Put(Bytes.toBytes(obj._1))
        put.addColumn(Bytes.toBytes(Source.basic.toString), Bytes.toBytes("value"), Bytes.toBytes(obj._2))
        (new ImmutableBytesWritable, put)
      })
      )
    val statsHbaseJobConf = HBaseUtil.getJobConf("profile_stats", zkQuorum, zkClientPort)
    hbaseStatsRdd.saveAsNewAPIHadoopDataset(statsHbaseJobConf)
  }


  def mkPut(rowKey : String, columnMap : Map[String, Any], acc : Accumulator[mutable.Map[String, Long]], profileType: ProfileType.Value,
            windows: Windows.Value, strBacktrackingDay : String) : Put = {
    var source = ""
    for ((k, v) <- columnMap) {
      val words = k.split("_")
      if (words.size >= 3) {
        source = words(0)
        //把所有类型转换为s，放入到acc中，acc的value是Long，但是在saveStatsToHbase的时候会被转换为string
        //val dataType = "s"
        val columnName = words.toList.slice(2, words.size).mkString("_")
        //val concatColumnName = "%s_%s_%s_%s".format(profileType.toString, windows.toString, strBacktrackingDay, columnName)
        acc.add(mutable.Map[String, Long](
          "%s_%s_%s_%s_%s".format(profileType.toString, source, windows.toString, strBacktrackingDay, columnName) -> 1l
          //"%s_%s_%s_%s".format(source, dataType, concatColumnName, CalType.count.toString) -> 1l
        ))
      }
    }
    return HBaseUtil.mkPut(rowKey, columnMap)
  }

  def getProfileType(key: String): ProfileType.Value = key match {
      case "smid" => ProfileType.smid;
      case "imei" => ProfileType.imei;
      case "idfa" => ProfileType.idfa;
      case "prcid" => ProfileType.prcid;
      case "phone" => ProfileType.phone;
      case "ip" => ProfileType.ip;
      case "token" => ProfileType.token;
      case "stats" => ProfileType.stats;
      case "smidbrother" => ProfileType.smidbrother;
  }
}

