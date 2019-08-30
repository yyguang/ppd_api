#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author fu jun hao <fujunhao@ishumei.com>
import sys
import os
import subprocess

def check_hdfs(path):
  rst = subprocess.Popen(['hdfs','dfs','-test','-e',path]).wait()
  if int(rst) == 0:
    print("path exists")
    print (subprocess.Popen(['hdfs','dfs','-rmr',path]).wait())
