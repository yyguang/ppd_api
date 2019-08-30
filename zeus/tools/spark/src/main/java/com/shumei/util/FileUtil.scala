package com.shumei.util

import java.io.Serializable

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path

class FileUtil  extends Serializable {

}

object FileUtil {
  def exist(filename : String) : Boolean = {
    val conf = new Configuration()
    val hdfsFileSytem = FileSystem.get(conf)
    hdfsFileSytem.exists(new Path(filename))
  }
}