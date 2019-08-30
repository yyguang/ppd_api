DATE1=$(date -d "1 days ago" +%Y%m%d)
DATE2=$(date +%Y%m%d)
DATE3=$(date -d "14 days ago" +%Y%m%d)
smoke_dir=$(cd `dirname $0`; pwd)
cd $smoke_dir
mkdir -p diff/$DATE1

sh ../getdiff.sh get_smoke.py $DATE1 >diff/$DATE1/spark.log 2>&1 &

wait
for i in shumei_smoke
do
hdfs dfs -cat diff/$DATE1/smoke/$i/part-00000|grep http|sort -k4nr|awk '{print $3}'|head -n800 >diff/$DATE1/$i.list
done
hdfs dfs -cat diff/$DATE1/smoke/shumei_smoke/part-00000|grep http|sort -k4nr|awk '{print $3}'|head -n200 >diff/$DATE1/shumei_smoke.list

#hdfs dfs -cat diff/$DATE1/tupu_smoke/part-00000|grep http|sort -k5nr|awk '{print $3}'|head -n200 >diff/$DATE1/tupu_smoke.list
#wait
sh download_daily_smoke.sh $1
