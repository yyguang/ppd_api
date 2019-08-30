package com.shumei.util

import java.io.Serializable

import org.apache.spark.{Accumulator, AccumulatorParam, SparkContext}

import scala.collection.mutable

class MapAccumulator extends AccumulatorParam[mutable.Map[String, Long]] {
  override def addInPlace(r1: mutable.Map[String, Long], r2: mutable.Map[String, Long]) : mutable.Map[String, Long] = {
    return mergeMap(r1, r2)
  }

  override def zero(initialValue: mutable.Map[String, Long]) : mutable.Map[String, Long] = {
    return mutable.Map[String, Long]()
  }

  def mergeMap(map1 : mutable.Map[String, Long], map2 : mutable.Map[String, Long]) : mutable.Map[String, Long] = {
    for ((k, v) <- map2) {
      var value = map1.getOrElse(k, 0l)
      value += v
      map1.put(k, value)
    }
    return map1
  }
}

object MapAccumulatorUtil extends Serializable {
  def getMapAccumulator(sc : SparkContext): Accumulator[mutable.Map[String, Long]] = {
    return sc.accumulator(mutable.Map[String, Long]())(new MapAccumulator())
  }
}
