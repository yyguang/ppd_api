DATE1=$(date -d "1 days ago" +%Y%m%d)
DATE2=$(date +%Y%m%d)
DATE3=$(date -d "14 days ago" +%Y%m%d)
ad_dir=$(cd `dirname $0`; pwd)
cd $ad_dir
mkdir -p daily_pics/$DATE1


wait

for i in shumei_dianshang shumei_jupaizi shumei_shouxiedazi shumei_shouxieti shumei_weixin shumei_lianxifangshishuiyin
do
mkdir daily_pics/$DATE1/$i

python ../download_insert.py  diff/$DATE1/$i.list daily_pics/$DATE1/$i ad>diff/$DATE1/download_$i.log 2>&1 

done
wait
#scp -r daily_pics/$DATE1 imgupload@admin1.bj.sm:/mnt/imgupload/daily_ad/$DATE1
rsync -av --delete daily_pics/$DATE1 imgupload@admin1.bj.sm:/mnt/imgupload/daily_ad/
python ../hash_sort.py daily_pics/$DATE1 ad >diff/phash.list

for i in dianshang weixin
do
ls daily_pics/$DATE1/shumei_$i|head -n300|awk '{print "https://data.fengkongcloud.com/image/daily_ad/'$DATE1'/shumei_'$i'/"$1}' >diff/$DATE1/final_shumei_$i"_url".list
cat diff/phash.list|grep $i >diff/$DATE1/"final_shumei_"$i"_url".list
cat diff/$DATE1/final_shumei_$i"_url".list|head -n300|awk '{print $1",'$i'"}' >diff/$DATE1/shumei_$i"_task".list
curl --data-binary "@diff/$DATE1/shumei_"$i"_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG&taskType=category-general-ad" >diff/$DATE1/shumei_$i"_task"_list.list
done

for i in shouxieti
do
ls daily_pics/$DATE1/shumei_$i|head -n400|awk '{print "https://data.fengkongcloud.com/image/daily_ad/'$DATE1'/shumei_'$i'/"$1}' >diff/$DATE1/final_shumei_$i"_url".list
cat diff/phash.list|grep $i >diff/$DATE1/"final_shumei_"$i"_url".list
cat diff/$DATE1/final_shumei_$i"_url".list|head -n400|awk '{print $1",'$i'"}' >diff/$DATE1/shumei_$i"_task".list
curl --data-binary "@diff/$DATE1/shumei_"$i"_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG&taskType=category-general-ad" >diff/$DATE1/shumei_$i"_task"_list.list
done

for i in lianxifangshishuiyin
do
ls daily_pics/$DATE1/shumei_$i|head -n400|awk '{print "https://data.fengkongcloud.com/image/daily_ad/'$DATE1'/shumei_'$i'/"$1}' >diff/$DATE1/final_shumei_$i"_url".list
cat diff/phash.list|grep $i >diff/$DATE1/"final_shumei_"$i"_url".list
cat diff/$DATE1/final_shumei_$i"_url".list|head -n400|awk '{print $1",lianxifangshishuiyin"}' >diff/$DATE1/shumei_$i"_task".list
curl --data-binary "@diff/$DATE1/shumei_"$i"_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG&taskType=category-general-ad" >diff/$DATE1/shumei_$i"_task"_list.list
done

for i in shouxiedazi
do
ls daily_pics/$DATE1/shumei_$i|head -n500|awk '{print "https://data.fengkongcloud.com/image/daily_ad/'$DATE1'/shumei_'$i'/"$1}' >diff/$DATE1/final_shumei_$i"_url".list
cat diff/phash.list|grep $i >diff/$DATE1/"final_shumei_"$i"_url".list
cat diff/$DATE1/final_shumei_$i"_url".list|head -n500|awk '{print $1",'$i'"}' >diff/$DATE1/shumei_$i"_task".list
curl --data-binary "@diff/$DATE1/shumei_"$i"_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG&taskType=category-general-ad" >diff/$DATE1/shumei_$i"_task"_list.list
done

for i in jupaizi
do
ls daily_pics/$DATE1/shumei_$i|head -n200|awk '{print "https://data.fengkongcloud.com/image/daily_ad/'$DATE1'/shumei_'$i'/"$1}' >diff/$DATE1/final_shumei_$i"_url".list
cat diff/phash.list|grep $i >ad/$DATE1/"final_shumei_"$i"_url".list
cat diff/$DATE1/final_shumei_$i"_url".list|head -n200|awk '{print $1",'$i'"}' >diff/$DATE1/shumei_$i"_task".list
curl --data-binary "@diff/$DATE1/shumei_"$i"_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG&taskType=category-general-ad" >diff/$DATE1/shumei_$i"_task"_list.list
done

python ../gen_mail.py diff/$DATE1 ad > $1
wait
