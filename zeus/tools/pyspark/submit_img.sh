#!/bin/bash
export PATH=/opt/anaconda3/bin:$PATH
export PYSPARK_PYTHON=/opt/anaconda3/bin/ipython
export PYSPARK_DRIVER_PYTHON=/opt/anaconda3/bin/ipython
export SPARK_HOME=/opt/spark-2.1.1-bin-hadoop2.6
#export LOCAL_CONF=/home/quanpinjie/new_prejectot/knowledge-graph
#export SPARK_CONF_DIR=$LOCAL_CONF/conf
export PYTHONHASHSEED=0
export PATH=$SPARK_HOME/bin:$PATH
dt=$1
out=$2
org=$3
#threshold=$4
spark-submit --master yarn --conf spark.port.maxRetries=2000 --conf spark.blockManager.port=1923 --executor-memory 1G --queue feature  catch_img_case.py ${dt} ${out} ${org}
