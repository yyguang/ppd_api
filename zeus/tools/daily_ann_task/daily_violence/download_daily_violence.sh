DATE1=$(date -d "1 days ago" +%Y%m%d)
DATE2=$(date +%Y%m%d)
DATE3=$(date -d "14 days ago" +%Y%m%d)

violence_dir=$(cd `dirname $0`; pwd)
cd $violence_dir
mkdir -p daily_pics/$DATE1


wait

#for i in shumei_kongbuzuzhi shumei_zhengchangzongjiao shumei_baoluanchangjing shumei_guoqiguohui shumei_junzhuang shumei_qiangzhidaoju shumei_xuexingchangjing tupu_terror shumei_ertongxiedian shumei_youxiqiangzhidaoju
for i in shumei_baoluanchangjing shumei_ertongxiedian shumei_xuexingchangjing shumei_anheidongman
do
mkdir daily_pics/$DATE1/$i

python ../download_insert.py  diff/$DATE1/$i.list daily_pics/$DATE1/$i violence >diff/$DATE1/download_$i.log 2>&1 

done
wait
#scp -r daily_pics/$DATE1 imgupload@admin1.bj.sm:/mnt/imgupload/daily_violence/$DATE1
rsync -av --delete daily_pics/$DATE1 imgupload@admin1.bj.sm:/mnt/imgupload/daily_violence/

python ../hash_sort.py daily_pics/$DATE1 violence >diff/phash.list

#for i in guoqiguohui junzhuang zhengchangzongjiao
for i in ertongxiedian baoluanchangjing xuexingchangjing anheidongman
do
ls daily_pics/$DATE1/"shumei_"$i|head -n300|awk '{print "https://data.fengkongcloud.com/image/daily_violence/'$DATE1'/shumei_'$i'/"$1}' >diff/$DATE1/final_shumei_$i"_url".list
cat diff/phash.list|grep shumei_$i >diff/$DATE1/"final_shumei_"$i"_url".list

cat diff/$DATE1/final_shumei_$i"_url".list|head -n200|awk '{print $1",'$i'"}' >diff/$DATE1/shumei_$i"_task".list
curl --data-binary "@diff/$DATE1/shumei_"$i"_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG&taskType=category-violence" >diff/$DATE1/shumei_$i"_task"_list.list
done


#for i in kongbuzuzhi baoluanchangjing qiangzhidaoju xuexingchangjing ertongxiedian youxiqiangzhidaoju
#do
#ls daily_pics/$DATE1/shumei_$i|head -n200|awk '{print "https://data.fengkongcloud.com/image/daily_violence/'$DATE1'/shumei_'$i'/"$1}' >diff/$DATE1/final_shumei_$i"_url".list
#cat diff/phash.list|grep $i >diff/$DATE1/"final_shumei_"$i"_url".list
#cat diff/$DATE1/final_shumei_$i"_url".list|head -n200|awk '{print $1",zhengchang"}' >diff/$DATE1/shumei_$i"_task".list
#curl --data-binary "@diff/$DATE1/shumei_"$i"_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG&taskType=category-violence" >diff/$DATE1/shumei_$i"_task"_list.list
#done



#ls daily_pics/$DATE1/tupu_terror|head -n200|awk '{print "https://data.fengkongcloud.com/image/daily_violence/'$DATE1'/tupu_terror/"$1}' >diff/$DATE1/final_tupu_terror_url.list
#cat diff/phash.list|grep tupu_terror|head -n200 >diff/$DATE1/final_tupu_terror_url.list
#cat diff/$DATE1/final_tupu_terror_url.list|awk '{print $1",zhengchang"}' >diff/$DATE1/tupu_terror_task.list
#curl --data-binary "@diff/$DATE1/tupu_terror_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG&taskType=category-violence" >diff/$DATE1/tupu_terror_task_list.list

python ../gen_mail.py diff/$DATE1 violence > $1
wait

rm shumei_100k
