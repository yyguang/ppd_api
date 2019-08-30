package com.shumei.util
import com.shumei.config.ShumeiConfig
import java.text.SimpleDateFormat
import java.util.{Calendar, Date, Locale, TimeZone}

import scala.collection.mutable.ArrayBuffer



class TimeUtil extends Serializable {

}

object TimeUtil {
  /**
    * Return list time, from (startDate = endDate - stepInSec * backTrackingStepCount) to (endDate)
    * Example: endDate = 20170927, stepInMillSec = 1000 * 3600 * 24, backTrackingStepCount = 7, format = yyyyMMdd
    * Then return Array[String] {20170920, 20170921, 20170922, 20170923, 20170924, 20170925, 20170926, 20170927}
    * Becareful return result size is backTrackingStepCount + 1
    * @param endDate
    * @param stepInMillSec
    * @param backTrackingStepCount
    * @param format
    * @return
    */
  def getStrListTime(endDate : String, stepInMillSec : Long, backTrackingStepCount : Int, format : String) : Array[String] = {
    //val format = new SimpleDateFormat("yyyyMMdd")
    val f = new SimpleDateFormat(format)
    val end = f.parse(endDate.toString).getTime
    val start = end - stepInMillSec * backTrackingStepCount
    var tmp = start
    val result = ArrayBuffer[String]()
    while (tmp <= end) {
      val date = new Date(tmp)
      result.append(f.format(date))
      tmp += stepInMillSec
    }
    return result.toArray
  }

  /**
    * 得到timestamp
    * @param timestamp
    * @return
    */
  def floorDayTimestamp(timestamp : Long) : Long = {
    val date = new Date(timestamp)
    val cal = Calendar.getInstance
    cal.setTimeZone(TimeZone.getTimeZone("GMT+8"))
    cal.setTime(date)
    cal.set(Calendar.HOUR_OF_DAY, 0)
    cal.set(Calendar.MINUTE, 0)
    cal.set(Calendar.SECOND, 0)
    cal.set(Calendar.MILLISECOND, 0)
    return cal.getTime.getTime
    // East Area 8
    //return (timestamp + 1000 * 3600 * 8) / 1000 / 3600 / 24 * 1000 * 3600 * 24
  }


  def dayCount(timestamp : Long) : Long = {
    return timestamp / (1000 * 3600 * 24)
  }

  def floorHourTimestamp(timestamp : Long) : Long = {
    return timestamp / (1000 * 3600) * (1000 * 3600)
  }



  /**
    *
    * @param timestamp
    * @return
    */
  def isInWorkTime(timestamp : Long) : Boolean = {
    val date = new Date(timestamp)
    val cal = Calendar.getInstance
    cal.setTimeZone(TimeZone.getTimeZone("GMT+8"))
    cal.setTime(date)
    val dayOfWeek = cal.get(Calendar.DAY_OF_WEEK)
    // 周一是一周的第二天
    if (dayOfWeek >= ShumeiConfig.workStartDayOfWeek && dayOfWeek <= ShumeiConfig.workEndDayOfWeek) {
      val hourOfDay = cal.get(Calendar.HOUR_OF_DAY)
      if (hourOfDay >= ShumeiConfig.workStartHourOfDay && hourOfDay <= ShumeiConfig.workEndHourOfDay) {
        return true
      }
    }
    return false
  }

  def isInRestTime(timestamp: Long) : Boolean = {
    val date = new Date(timestamp)
    val cal = Calendar.getInstance
    cal.setTimeZone(TimeZone.getTimeZone("GMT+8"))
    cal.setTime(date)
    val hourOfDay = cal.get(Calendar.HOUR_OF_DAY)
    if ((hourOfDay >= 0 && hourOfDay <= ShumeiConfig.restEndHourOfDay) ||
      (hourOfDay >= ShumeiConfig.restStartHourOfDay && hourOfDay <= 24)) {
      return true
    }
    return false
  }

  /**
    * 是否在凌晨
    * @param timestamp
    * @return
    */
  def isInSmallHours(timestamp: Long) : Boolean = {
    val date = new Date(timestamp)
    val cal = Calendar.getInstance
    cal.setTimeZone(TimeZone.getTimeZone("GMT+8"))
    cal.setTime(date)
    val hourOfDay = cal.get(Calendar.HOUR_OF_DAY)
    if ((hourOfDay >= 0 && hourOfDay <= 5)) {
      true
    } else {
      false
    }
  }

  /**
    * 把可能是s的时间转换为毫秒
    * @param t
    * @return
    */
  def timeToTimestamp(t : Long) : Long = {
    if (t < 1000000000000l){
      t * 1000
    } else {
      t
    }
  }

  /**
    * 把yyyy-MM-dd hh:mm:ss的时间转换为毫秒
    * @param s
    * @return
    */
  def strToTimestamp(s : String) : Long = {
    val sdf = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss")
    val d = sdf.parse(s)
    return d.getTime
  }

  def timestampToStr(ts : Long, format : String) : String = {
    val sdf = new SimpleDateFormat(format)
    sdf.setTimeZone(TimeZone.getTimeZone("GMT+8"))
    return sdf.format(ts)
  }

  /**
    * 把窗口时间30,60,90,all等转换为30d,60d,90d,all等
    * @param s
    * @return
    */
  def formatWindowsDayStr(s : String): String = {
    if (s.equals("all")) {
      return "all"
    } else {
      return "%sd".format(s)
    }
  }

  def genBacktrackingDay(ts : Long) : String = {
    return timestampToStr(ts, "yyyyMMdd")
  }
}
