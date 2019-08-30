// Author Tony Tsao <cao-teng-teng.cao@foxmail.com> 2017-12-13 11:28:43
package com.shumei.profile.app

import com.shumei.util.{JacksonUtil, TimeUtil, CsvUtil}
import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import com.shumei.util.JsonSmartUtil
import org.json4s._
import org.json4s.jackson.JsonMethods._

class GetAppList {

}

object GetAppList {
  def jsonGet(a: Object, key: String) = JsonSmartUtil.getValueFromObjectBySpecKey(a, key)

  def getList(data : Map[String, Any], key : String): List[Any] = {
    val value = data.getOrElse(key, Map[String, Any]())
    if (value.equals("")) {
      return List[Any]()
    }
    return value.asInstanceOf[List[Any]]
  }

  // args: fpPath, outputPath, outputPartitions
  //       fp的路径，输出的目录路径，输出的csv文件gz压缩成多少分卷
  // 风险app挖掘：输出CSV按appId进行合并，三列定义为：
  // numOfApp, appId, appNames
  // 出现的次数，app的ID，app的名称列表（去重）
  def main(args : Array[String]): Unit = {
    val conf = new SparkConf().setAppName("com.shumei.app.GetAppList")
    val sc = new SparkContext(conf)

    //val fpPath = "head10.txt"
    //val outputPath = "GetAppList_output"
    //val outputPathNumPartitions = 1

    val fpPath = args(0)
    val outputPath = args(1)
    val outputPartitions = args(2).toInt

    val fpRdd = sc.textFile(fpPath)
    val appRdd = fpRdd
    .flatMap(line => {
      try {
        val result = ArrayBuffer[Tuple2[String, String]]()
        val root = JsonSmartUtil.loads(line)
        val apps1 = jsonGet(root, "data@@@apps") 
        if(apps1 != null) {
            val apps =  parse(apps1.toString).values.asInstanceOf[List[String]]
            for (app <- apps) {
                // 标准格式为安装时间戳,appid,appChName,是否用户APP，旧版本SDK只有前两项
                val words = app.split(",")
                if (words.size > 1) {
                    val appId = words(1)
                    val appChName = if(words.size > 3) words(2) else ""
                    result.append((appId, appChName))
                }
            }
            if (result.size > 0) result.toList else None
        }else {
            None
        }

      } catch {
        case _ : Exception => None
      }
    })
    .combineByKey(
      x => if (x == "") (Set.empty[String], 1) else (Set(x), 1), 
      (x:(Set[String], Int), y:String) => if (y == "") (x._1, x._2 + 1) else (x._1 + y, x._2 + 1), 
      (x:(Set[String], Int), y:(Set[String], Int)) => (x._1 ++ y._1, x._2 + y._2)
    )
    .map(t => Array(t._2._2.toString, t._1, t._2._1.mkString(",")))
    //在这个combineByKey中，可以看到首先每次遇到第一个值，就将其变为一个加入到一个Set中去。
    //第二个函数指的是在key相同的情况下，当每次遇到新的value值，就把这个值添加到这个Set中去。
    //最后是一个merge函数，表示将key相同的两个Set进行合并。
    
    //appRdd.collect().foreach {println}
    CsvUtil.saveCsv(appRdd, outputPath, outputPartitions)
  }
}
