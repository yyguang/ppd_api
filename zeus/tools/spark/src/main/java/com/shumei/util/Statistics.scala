package com.shumei.util

import java.io.Serializable


class Statistics(val data : Array[Double]) extends Serializable {
  val size = data.length

  def getMean() : Double = {
    if (size == 0) {
      0f
    } else {
      (data.sum / size).formatted("%.4f").toDouble
    }
  }

  def getCount() : Int = {
    size
  }

  def getVariance() : Double = {
    val mean = getMean()
    var temp = 0d
    for (a <- data) {
      temp += (a - mean) * (a - mean)
    }
    if (size == 0) {
      0
    } else {
      (temp / size).formatted("%.4f").toDouble
    }
  }

  def getStdDev() : Double = {
    return (Math.sqrt(getVariance())).formatted("%.4f").toDouble
  }

  def getMedian() : Double = {
    if (size != 0) {
      val sortData = data.sorted

      if (sortData.length % 2 == 0) {
        (sortData.apply((sortData.length / 2) - 1) + sortData.apply(sortData.length / 2)) / 2.0
      } else {
        sortData(sortData.length / 2)
      }
    } else {
      0
    }
  }

  def getSum() : Double = {
    return data.sum
  }

  def getMax() : Double = {
    return data.max
  }

  def getMin() : Double = {
    return data.min
  }
}
