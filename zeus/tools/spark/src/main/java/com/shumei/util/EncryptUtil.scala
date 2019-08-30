package com.shumei.util

import org.apache.commons.lang.exception.ExceptionUtils

class EncryptUtil extends java.io.Serializable {

}

object EncryptUtil {
  def encrypt(url : String, content : String): String = {
    try {
      return HttpUtil.get(url + content)
    } catch {
      case ex : Exception => {
        println(ExceptionUtils.getFullStackTrace(ex))
        return ""
      }
    }

  }
}
