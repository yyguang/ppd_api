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
special=$3
sample=$4
#随机生成端口号
port=$(shuf -i 1234-65000 -n 1)
spark-submit --master yarn --executor-memory 1G --queue feature  --conf spark.blockManager.port=$port catch_img_case.py ${dt} ${out} ${special} ${sample}
hdfs dfs -get result/${out}/part-00000
