DATE1=$(date -d "1 days ago" +%Y%m%d)
DATE2=$(date +%Y%m%d)
DATE3=$(date -d "14 days ago" +%Y%m%d)
smoke_dir=$(cd `dirname $0`; pwd)

cd $smoke_dir
mkdir -p daily_pics/$DATE1


wait

for i in shumei_smoke
do
mkdir daily_pics/$DATE1/$i

python ../download_insert.py  diff/$DATE1/$i.list daily_pics/$DATE1/$i smoke >diff/$DATE1/download_$i.log 2>&1 

done
wait
#scp -r daily_pics/$DATE1 imgupload@admin1.bj.sm:/mnt/imgupload/daily_smoke/$DATE1
rsync -av --delete daily_pics/$DATE1 imgupload@admin1.bj.sm:/mnt/imgupload/daily_smoke/

python ../hash_sort.py daily_pics/$DATE1 smoke >diff/phash.list

ls daily_pics/$DATE1/shumei_smoke|head -n300|awk '{print "https://data.fengkongcloud.com/image/daily_smoke/'$DATE1'/shumei_smoke/"$1}' >diff/$DATE1/final_shumei_smoke_url.list
cat diff/phash.list|grep shumei_smoke|head -n300 >diff/$DATE1/final_shumei_smoke_url.list

cat diff/$DATE1/final_shumei_smoke_url.list|awk '{print $1",normal"}' >diff/$DATE1/shumei_smoke_task.list
curl --data-binary "@diff/$DATE1/shumei_smoke_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG&taskType=category-smoke" >diff/$DATE1/shumei_smoke_task_list.list

#ls smoke_pics/$DATE1/tupu_smoke|head -n300|awk '{print "https://data.fengkongcloud.com/image/daily_smoke/'$DATE1'/tupu_smoke/"$1}' >smoke/$DATE1/final_tupu_smoke_url.list
#cat smoke/phash.list|grep tupu_smoke|head -n300 >smoke/$DATE1/final_tupu_smoke_url.list
#cat smoke/$DATE1/final_tupu_smoke_url.list|awk '{print $1",normal"}' >smoke/$DATE1/tupu_smoke_task.list
#curl --data-binary "@smoke/$DATE1/tupu_smoke_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG&taskType=category-smoke" >smoke/$DATE1/tupu_smoke_task_list.list
python ../gen_mail.py diff/$DATE1 smoke > $1

