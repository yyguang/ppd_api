package com.shumei.util

import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.{SparkConf, SparkContext}
import au.com.bytecode.opencsv.CSVWriter
import java.io.StringWriter
import scala.collection.JavaConversions._
import java.io.StringReader
import au.com.bytecode.opencsv.CSVReader
import org.apache.spark.rdd.RDD

class CsvUtil {

}

object CsvUtil {
  def saveCsv(rdd: RDD[Array[String]], path: String, partitions: Int): Unit = {
    val toCsv = (a: Array[String]) => {
      val buf = new StringWriter
      val writer = new CSVWriter(buf)
      writer.writeAll(List(a))
      buf.toString.trim
    }
    rdd.map(a => toCsv(a))
    .repartition(partitions)
    .saveAsTextFile(path)
  }
  
  def loadCsv(sc: SparkContext, path: String): RDD[Array[String]] = {
    val input = sc.textFile(path)
    val rdd = input
    .map(line => {
      val reader = new CSVReader(new StringReader(line));
      reader.readNext()
    })
    rdd
  }
}
