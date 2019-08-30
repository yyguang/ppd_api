package com.shumei.util

import org.apache.commons.lang.exception.ExceptionUtils

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import java.math.BigInteger
import java.security.MessageDigest

class CommonUtil extends Serializable {

}

object CommonUtil {
  def isEnctype(s : String): Boolean = {
    if (s.contains("A") || s.contains("B") || s.contains("C") ||
      s.contains("D") || s.contains("E") || s.contains("F")) {
      return true
    }
    return false
  }

  def printErrorByErrorInterval(errorCount : Long, printErrorInterval : Long,
                               e : Exception, msg : String): Unit = {
    if (errorCount % printErrorInterval == 0) {
      println(msg)
      println(ExceptionUtils.getFullStackTrace(e))
    }
  }

  /**
    * 从obj里面获取specKey的value，其中specKey类似于A@@@B@@@C，表示获取A -> B -> C的value
    * @param obj
    * @param specKey
    */
  def getValueFromMapBySpecKey(obj : Map[String, Any], specKey : String): Option[Any] = {
    val words = specKey.split("@@@")
    val mapKeyWords = words.slice(0, words.length - 1)
    val realKey = words(words.length - 1)
    var tmpObj = obj
    for (word <- mapKeyWords) {
      tmpObj = tmpObj.getOrElse(word, Map[String, Any]()).asInstanceOf[Map[String, Any]]
    }
    return tmpObj.get(realKey)
  }

  def getValueFromMapBySpecKeyByType[T](obj : Map[String, Any], specKey : String) : Option[T] = {
    val value = getValueFromMapBySpecKey(obj, specKey)
    if (value == None) {
      return None
    } else {
      return Some(value.get.asInstanceOf[T])
    }
  }

  /**
    * 把多层map转换为只有一层的map，只获取以specKeyList中值为key的value
    * @param obj
    * @param specKeyList
    * @return
    */
  def flatMapBySpecKeyList(obj : Map[String, Any], specKeyList : List[String]) : Map[String, Any] = {
    val resultMap = mutable.Map[String, Any]()
    for (specKey <- specKeyList) {
      val value = getValueFromMapBySpecKey(obj, specKey)
      if (value != None) {
        resultMap.put(specKey, value.get)
      }
    }
    return resultMap.toMap
  }


  /**
    *
    * @param b
    * @return
    */
  def booltoInt(b:Boolean) = if (b) 1 else 0

  def trySetMap(key : String, fromMap : Map[String, Any], newKey : String, toMap : mutable.Map[String, Any]): Unit = {
    val opt = fromMap.get(key)
    if (opt != None) {
      toMap.put(newKey, opt.get)
    }
  }

  def anyArrayToDoubleArray(a : ArrayBuffer[Any]) : ArrayBuffer[Double] = {
    val resultArray = ArrayBuffer[Double]()
    for (value <- a) {
      resultArray.append(value.asInstanceOf[Number].doubleValue())
    }
    return resultArray
  }

  def biggerThanOrEqualCount(a : Array[Double], stdValue : Double): Int = {
    var count = 0
    for (value <- a) {
      if (value >= stdValue) {
        count += 1
      }
    }
    return count
  }

  def md5(s: String) = try {
    val md = MessageDigest.getInstance("MD5")
    val bytes = md.digest(s.getBytes("utf-8"))
    toHex(bytes).toLowerCase
  } catch {
    case e: Exception =>
      throw new RuntimeException(e)
  }

  private def toHex(bytes: Array[Byte]) = {
    val HEX_DIGITS = "0123456789ABCDEF".toCharArray
    val ret = new StringBuilder(bytes.length * 2)
    var i = 0
    while ( {
      i < bytes.length
    }) {
      ret.append(HEX_DIGITS((bytes(i) >> 4) & 0x0f))
      ret.append(HEX_DIGITS(bytes(i) & 0x0f))

      {
        i += 1; i - 1
      }
    }
    ret.toString
  }

  def addRelationMap(relationMap : mutable.Map[String, Long], key : String, value : Long): Unit = {
    val ts = relationMap.getOrElse(key, 0l)
    if (ts == 0l || ts < value) {
      relationMap.put(key, value)
    }
  }
}