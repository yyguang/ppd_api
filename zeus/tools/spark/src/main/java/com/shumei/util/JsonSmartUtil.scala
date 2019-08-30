package com.shumei.util

import java.io.Serializable

import net.minidev.json.{JSONArray, JSONObject}
import net.minidev.json.parser.JSONParser
import com.shumei.util.JacksonUtil

class JsonSmartUtil extends Serializable {

}

object JsonSmartUtil {

  def loads(line : String) : Object = {
    val parser = new JSONParser
    return parser.parse(line)
  }

  def dumps(obj : Object): String = {
    return JSONObject.toJSONString(obj.asInstanceOf[JSONObject])
  }

  /**
    * 从obj里面获取specKey的value，其中specKey类似于A@@@B@@@C，表示获取A -> B -> C的value
    * @param obj
    * @param specKey
    */
  def getValueFromObjectBySpecKey(obj : Object,  specKey : String): Object = {
    val words = specKey.split("@@@")
    val mapKeyWords = words.slice(0, words.length - 1)
    val realKey = words(words.length - 1)
    var tmpObj = obj
    try {
        for (word <- mapKeyWords) {
            val tmpVal = tmpObj.asInstanceOf[JSONObject].get(word)
            if(tmpVal == null) {
                return null
            }
            tmpObj = tmpVal
        }
        return tmpObj.asInstanceOf[JSONObject].get(realKey)
    } catch {
        case e: Exception => {
            return null
        }
    }
  }

  /**
    * 从obj里面获取specKey的value，其中specKey类似于A@@@B@@@C，表示获取A -> B -> C的value
    * @param obj
    * @param specKey
    */
  def getValueFromObjectBySpecKeyByAny(obj : Any,  specKey : String): Any = {
    val words = specKey.split("@@@")
    val mapKeyWords = words.slice(0, words.length - 1)
    val realKey = words(words.length - 1)
    var tmpObj = obj
    try {
      for (word <- mapKeyWords) {
        val tmpVal = tmpObj.asInstanceOf[JSONObject].get(word)
        if(tmpVal == null) {
          return null
        }
        tmpObj = tmpVal
      }
      return tmpObj.asInstanceOf[JSONObject].get(realKey)
    } catch {
      case e: Exception => {
        return null
      }
    }
  }

  def getValueFromObjectBySpecKeyByType[T](obj : Any, specKey : String) : Option[T] = {
    val value = getValueFromObjectBySpecKeyByAny(obj, specKey)
    if (value == null) {
      return None
    } else {
      return Some(value.asInstanceOf[T])
    }
  }


  def ConvertObjToScalaObj[T](obj : Object, classType : Class[T]) : T = {
    return JacksonUtil.loads(obj.toString, classType)
  }

//  import scala.collection.JavaConversions._
//  for (k <- featureMap.entrySet()) {
//    println(k.getKey())
//    println(k.getValue)
//  }
//
//  str_list.toArray()
}
