package com.shumei.util

import java.io.Serializable
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule

class JacksonUtil extends Serializable {

}

object JacksonUtil extends Serializable {
  val mapper = new ObjectMapper
  mapper.registerModule(DefaultScalaModule)

  def loads[T](line : String, classType : Class[T]) : T = {
    val obj = mapper.readValue(line, classType)
    return obj
  }

  def dumps(obj : Any): String = {
    return mapper.writeValueAsString(obj)
  }
}
