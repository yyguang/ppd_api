// Author Tony Tsao <cao-teng-teng.cao@foxmail.com> 2017-12-13 11:28:43
package com.shumei.profile.app

import org.apache.spark.{SparkConf, SparkContext}
import com.shumei.util.{JsonSmartUtil, TimeUtil, CsvUtil, CommonUtil, HBaseUtil}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import com.shumei.util.HBaseUtil.{ProfileType, Windows}
import org.apache.hadoop.hbase.io.ImmutableBytesWritable

class MergePackage2Category {

}

object MergePackage2Category {
  // 主函数
  def main(args : Array[String]): Unit = {
    val conf = new SparkConf().setAppName("com.shumei.profile.app.MergePackage2Category")
    val sc = new SparkContext(conf)

    val package2CatPath = "package2Category-20171208.csv"
    val platform2CatPath = "platform2Category-20171208.csv"
    val outputPath = "result/package2Cat"
    val outputPartitions = 1
    
    val rddPackage2Cat = CsvUtil.loadCsv(sc, package2CatPath)
    val rddPlatform2Cat = CsvUtil.loadCsv(sc, platform2CatPath)
  
    val platform2CatBuf = mutable.Map[String, String]()
    val platform2CatAll = rddPlatform2Cat.collect()
    for (item <- platform2CatAll)
      platform2CatBuf += (item(0).trim() -> item(1).trim())
    val platform2Cat = platform2CatBuf.toMap

    // 统计行数
    /*val rddResult = rddPackage2Cat.map(line => {
      val names = line(1).split(",")
      var cat = ""
      for (name <- names) {
        val name2 = name.trim()
        if (platform2Cat.contains(name2))
          cat = platform2Cat(name2)
      }
      if (cat != "" && line(3) != "") Some(line(0), cat) else None
    })
    .filter(x => x != None)*/

    // 合并到package
    val rddResult = rddPackage2Cat.map(line => {
      val names = line(1).split(",")
      var cat = ""
      for (name <- names) {
        val name2 = name.trim()
        if (platform2Cat.contains(name2))
          cat = platform2Cat(name2)
      }
      if (cat != "")
        line(3) = cat
      line
    })

    CsvUtil.saveCsv(rddResult, outputPath, outputPartitions)
  }
}
