package com.shumei.util

import scala.collection.mutable.ArrayBuffer

class FormatUtil {

}

object FormatUtil extends Serializable {
  def libSVMFormat(result : Map[String, Any], labelKey : String, formatMapConfig : Map[String, Int]): String = {
    val buf = ArrayBuffer[String]()

    val label = result.get(labelKey).get.toString
    buf.append(label)

    for ((key, index) <- formatMapConfig) {
      val value = result.get(key)
      if (value != None) {
        if (value.get.isInstanceOf[ArrayBuffer[Any]]) {
          buf.append("%d:%s".format(index, value.get.asInstanceOf[ArrayBuffer[Any]].mkString(",")))
        } else {
          buf.append("%d:%s".format(index, value.get.toString))
        }

      }
    }

    return buf.mkString(" ")
  }
}
