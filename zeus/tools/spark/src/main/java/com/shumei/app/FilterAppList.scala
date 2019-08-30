// Author Tony Tsao <cao-teng-teng.cao@foxmail.com> 2017-12-13 11:28:43
package com.shumei.profile.app

import com.shumei.util.{JacksonUtil, TimeUtil, CsvUtil}
import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class FilterAppList {

}

object FilterAppList {
  def main(args : Array[String]): Unit = {
    val conf = new SparkConf().setAppName("com.shumei.app.FilterAppList")
    val sc = new SparkContext(conf)
    
    val appPath = args(0)
    val categoryPath = args(1)
    val keptCategories = args(2).split(",")
    val outputPath = args(3)
    val outputPartitions = args(4).toInt

    //val appPath = "/user/caoteng/result/AppList/dt=20171111/*.gz"
    //val categoryPath = "/user/caoteng/app/category_file"
    //val keptCategories = "loan,financial_planning".split(",") 
    //val outputPath = "/user/caoteng/result/AppList/dt=20171111/filtered"
    //val outputPartitions = 1

    val categoryInput = sc.textFile(categoryPath)
    val categoryRdd = categoryInput
    .flatMap(line => {
      try {
        val obj = JacksonUtil.loads(line, classOf[Map[String, Any]])
        val packageName = obj.get("packageName").get.toString
        val enCategoryName = obj.get("enCategoryName").get.toString
        Some((packageName, enCategoryName))
      } catch {
        case _ : Exception => None
      }
    })
    .filter(line => (keptCategories contains line._2))

    val categoryAll = categoryRdd.collect()
    val packageMap = mutable.Map[String, String]()
    val packageSet = mutable.Set[String]()
    for (info <- categoryAll) {
      val packageName = info._1
      val enCategoryName = info._2
      packageMap.put(packageName, enCategoryName)
      packageSet.add(packageName)
    }
    val bdPackageMap = packageMap.toMap
    val bdPackageSet = packageSet.toSet
    
    //categoryRdd.take(1000).foreach {println}
    
    val appRdd = CsvUtil.loadCsv(sc, appPath)
    .filter(line => ((line != null) && (line.length == 3) && (bdPackageSet contains line(1))))
    .map(line => Array(line(1), line(2), bdPackageMap(line(1))))
    
    //val tmpRdd = appRdd.collect()
    //for(result <- tmpRdd) {
    //  for(re <- result) {
    //    println(re)
    //  }
    //}
    CsvUtil.saveCsv(appRdd, outputPath, outputPartitions)
  }
}
