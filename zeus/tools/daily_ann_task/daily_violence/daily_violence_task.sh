DATE1=$(date -d "1 days ago" +%Y%m%d)
DATE2=$(date +%Y%m%d)
DATE3=$(date -d "14 days ago" +%Y%m%d)

violence_dir=$(cd `dirname $0`; pwd)
cd $violence_dir
mkdir -p  diff/$DATE1

sh ../getdiff.sh get_violence.py $DATE1 >diff/$DATE1/spark.log 2>&1 &

wait

for i in shumei_anheidongman shumei_baoluanchangjing shumei_xuexingchangjing shumei_ertongxiedian
do
hdfs dfs -cat diff/$DATE1/violence/$i/part-00000|grep http|sort -k4nr|awk '{print $3}'|head -n400 >diff/$DATE1/$i.list
done
#hdfs dfs -cat diff/$DATE1/violence/shumei_guoqiguohui/part-00000|grep http|sort -k4nr|awk '{print $3}'|head -n400 >diff/$DATE1/shumei_guoqiguohui.list

#hdfs dfs -cat diff/$DATE1/tupu_terror/part-00000|grep http|sort -k5nr|awk '{print $3}'|head -n400 >diff/$DATE1/tupu_terror.list

hdfs dfs -get diff/$DATE1/violence/shumei_100k_terror/part-00000 shumei_100k

wait
sh download_daily_violence.sh $1
