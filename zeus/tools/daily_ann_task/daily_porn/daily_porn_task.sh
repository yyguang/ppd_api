DATE1=$(date -d "1 days ago" +%Y%m%d)
DATE2=$(date +%Y%m%d)
DATE3=$(date -d "7 days ago" +%Y%m%d)
mail_output=$1

porn_dir=$(cd `dirname $0`; pwd)
cd $porn_dir

mkdir -p diff/$DATE1
sh ../getdiff.sh getdiff.py $DATE1 >diff/$DATE1/spark.log 2>&1 &


wait
for i  in normal sexy porn
do
hdfs dfs -cat diff/$DATE1/porn/shumei_$i/part-00000|grep http|grep RaI8hPdp9MjVY5OsMCqH|awk '{if($5<0.5&&$5>0.1)print $3}'  >diff/$DATE1/duiduikeji_$i.list
hdfs dfs -cat diff/$DATE1/porn/shumei_$i/part-00000|grep http|grep tWCjPsD6El5NFceULAUi|awk '{if($5>0.05)print $3}'  >diff/$DATE1/tongzhuo_$i.list
hdfs dfs -cat diff/$DATE1/porn/shumei_$i/part-00000|grep http|grep swrSeJvWILbkRKZU0Gvj|awk '{if($5>0.05)print $3}'  >diff/$DATE1/maimai_$i.list
#hdfs dfs -cat diff/$DATE1/porn/shumei_$i/part-00000|grep http|grep 4AibJzNblS5pqgfDzo5d|awk '{if($5>0.05)print $3}'  >diff/$DATE1/ss99_$i.list
#hdfs dfs -cat diff/$DATE1/porn/shumei_$i/part-00000|grep http|grep fNpgAvvtSnD3udorfJ71|awk '{if($5>0.05)print $3}'  >diff/$DATE1/tiantianlangrensha_$i.list
done

hdfs dfs -cat diff/$DATE1/porn/shumei_porn/part-00000|grep http|grep N8iOMP9FGlbuygwJCPMk|awk '{if($5>0.5)print $3}' >diff/$DATE1/vipkid.list
hdfs dfs -cat diff/$DATE1/porn/shumei_porn/part-00000|grep http|grep eR46sBuqF0fdw7KWFLYa|awk '{if($5>0.5)print $3}' |shuf -n 600 >diff/$DATE1/xiaohongshu.list
hdfs dfs -cat diff/$DATE1/porn/shumei_porn/part-00000|grep http|grep DeBPYksyeqaBlCVSZXos|awk '{if($5>0.5)print $3}' |shuf -n 600 >diff/$DATE1/xiongmaotv.list
hdfs dfs -cat diff/$DATE1/porn/shumei_porn/part-00000|grep http|grep z8T9p6PjPS40Gq0F8w3q|awk '{if($5>0.5)print $3}' >diff/$DATE1/rela.list
hdfs dfs -cat diff/$DATE1/porn/shumei_porn/part-00000|grep http|grep gXolGHWQ8Fu37Ddhdy69|awk '{if($5>0.8)print $3}' >diff/$DATE1/xiaoying.list
hdfs dfs -cat diff/$DATE1/porn/shumei_porn/part-00000|grep http|grep LchL9Nhw5xPSCV1YinKy|awk '{if($5>0.8)print $3}' >diff/$DATE1/uki.list
hdfs dfs -cat diff/$DATE1/porn/shumei_porn/part-00000|grep http|grep 1LSK8wl2uyBr7oblW2Gz|awk '{if($5>0.8)print $3}' >diff/$DATE1/yizhibo.list
hdfs dfs -cat diff/$DATE1/porn/shumei_porn/part-00000|grep http|grep va7Yyjngp1Wb0iSW9xy7|awk '{if($5>0.5)print $3}' >diff/$DATE1/huanqiubushou.list
wait

sh download_daily_porn.sh ${mail_output}


