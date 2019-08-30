package com.shumei.util

import com.shumei.config.ShumeiConfig
import org.apache.commons.net.util.SubnetUtils
import org.apache.spark.SparkFiles

import scala.reflect.io.File

class IpUtil extends Serializable {

}

object IpUtil {
  var isLoadIpData = false
  /**
    *
    * @param ip
    * @return [country, province, city, UNKNOWN, ISP]
    */
  def getIpLocation(ip: String): Array[String] = {
    if (isLoadIpData == false) {
      val ipDataPath = SparkFiles.get(ShumeiConfig.ipDataName)
      val f = File(ipDataPath)
      if (f.exists) {
        IP.enableFileWatch = true
        IP.load(ShumeiConfig.ipDataName)
        isLoadIpData = true
      }
    }

    if (isLoadIpData == true) {
      try {
        return IP.find(ip)
      } catch {
        case e : Exception => {
          return Array("exception", "", "", "", "")
        }
      }
    } else {
      return Array("ip_data_not_load", "", "", "", "")
    }
  }

  def getNetworkAddress(ip : String, netmask : Int): String = {
    val utils = new SubnetUtils(ip + "/" + netmask)
    return utils.getInfo().getNetworkAddress
  }

//
//  def main(args : Array[String]): Unit = {
//    getNetworkAddress("192.168.0.3", 16)
//    //println("hello")
//  }
}