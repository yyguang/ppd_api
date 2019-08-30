DATE1=$(date -d "1 days ago" +%Y%m%d)
DATE2=$(date +%Y%m%d)
DATE3=$(date -d "14 days ago" +%Y%m%d)
ad_dir=$(cd `dirname $0`; pwd)
cd $ad_dir
mkdir -p diff/$DATE1
sh ../getdiff.sh get_politics.py $DATE1 >diff/$DATE1/spark.log 2>&1 &
wait
hdfs dfs -cat diff/$DATE1/politics/part-00000|grep http|sort -k4nr|awk '{print $3}'|head -n100 >diff/$DATE1/politics.list
wait
sh download_daily_politics.sh $1
