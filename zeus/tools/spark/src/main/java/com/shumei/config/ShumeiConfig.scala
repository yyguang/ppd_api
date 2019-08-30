package com.shumei.config

import java.io.Serializable

class ShumeiConfig extends Serializable {

}

object ShumeiConfig {
  val aeFilePath = "/user/data/event/detail_ae/dt=%s/"
  val fpFilePath = "/user/data/event/detail_fp/dt=%s/"
  val smsFilePath = "/user/data/fin/event2/dt=%s/"

  val aeAccountLoginIpFilePath = "/user/shiyufeng/fin/ip/ae/dt=%s/ACCOUNT_LOGIN/"
  val aeAccountRegisterIpFilePath = "/user/shiyufeng/fin/ip/ae/dt=%s/ACCOUNT_REGISTER/"
  val smsIpFilePath = "/user/shiyufeng/fin/ip/sms/dt=%s/"
  val fpIpFilePath = "/user/shiyufeng/fin/ip/fp/dt=%s/"

  val allIpRawFeaturePath = Array(fpIpFilePath, smsIpFilePath, aeAccountRegisterIpFilePath, aeAccountLoginIpFilePath)
  // 最常用IP，topN
  val ipFeatureTopIpCount = 2
  // 最长在城市，topN
  val cityFeatureTopCityCount = 2

  val allIpFeaturePath = "/user/shiyufeng/fin/ip_features/dt=%s/"

  // 10点
  val workStartHourOfDay = 10
  // 17点
  val workEndHourOfDay = 17
  // 第2天，也就是星期一
  val workStartDayOfWeek = 2
  // 第6天，也就是星期五
  val workEndDayOfWeek = 6


  val restStartHourOfDay = 22
  val restEndHourOfDay = 8

  val frequentIpRatio = 0.5

  val ipDataName = "ip_data"

}
