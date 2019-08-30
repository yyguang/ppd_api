DATE1=$(date -d "1 days ago" +%Y%m%d)
DATE2=$(date +%Y%m%d)
mail_output=$1

porn_dir=$(cd `dirname $0`; pwd)
cd $porn_dir

mkdir -p daily_pics/$DATE1

wait

for i in duiduikeji_normal duiduikeji_sexy duiduikeji_porn tongzhuo_normal tongzhuo_sexy tongzhuo_porn maimai_normal maimai_sexy maimai_porn xiaohongshu xiongmaotv vipkid rela xiaoying uki yizhibo huanqiubushou
do
mkdir daily_pics/$DATE1/$i

python ../download_insert.py diff/$DATE1/$i.list daily_pics/$DATE1/$i porn>diff/$DATE1/download_$i.log 2>&1 

done
wait
#scp -r daily_pics/$DATE1 imgupload@admin1.bj.sm:/mnt/imgupload/daily_porn/$DATE1
rsync -av --delete daily_pics/$DATE1 imgupload@admin1.bj.sm:/mnt/imgupload/daily_porn/
python ../hash_sort.py daily_pics/$DATE1 porn > phash_porn.list


num=`cat phash_porn.list|wc -l`
if [ $num -gt 5000 ] ;then
  python shuf_in_sort.py phash_porn.list 5000
fi

wait 

for i in porn sexy normal 
do
for j in duiduikeji tongzhuo maimai
do
ls daily_pics/$DATE1/$j"_"$i|awk '{print "https://data.fengkongcloud.com/image/daily_porn/'$DATE1'/'$j'_'$i'/"$1}' >diff/$DATE1/final_$j"_"$i"_url".list
cat phash_porn.list|grep $j"_"$i >diff/$DATE1/final_$j"_"$i"_url".list

cat diff/$DATE1/final_$j"_"$i"_url".list|awk '{print $1",'$i'"}' >diff/$DATE1/$j"_"$i"_task".list

curl --data-binary "@diff/$DATE1/"$j"_"$i"_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG" >diff/$DATE1/$j"_"$i"_task"_list.list

done
done


ls daily_pics/$DATE1/vipkid|awk '{print "https://data.fengkongcloud.com/image/daily_porn/'$DATE1'/vipkid/"$1}' >diff/$DATE1/final_vipkid_url.list
cat phash_porn.list|grep vipkid >diff/$DATE1/final_vipkid_url.list
cat diff/$DATE1/final_vipkid_url.list|awk '{print $1",normal"}' >diff/$DATE1/vipkid_task.list
curl --data-binary "@diff/$DATE1/vipkid_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG" >diff/$DATE1/vipkid_task_list.list


ls daily_pics/$DATE1/rela|awk '{print "https://data.fengkongcloud.com/image/daily_porn/'$DATE1'/rela/"$1}' >diff/$DATE1/final_rela_url.list
cat phash_porn.list|grep rela >diff/$DATE1/final_rela_url.list
cat diff/$DATE1/final_rela_url.list|awk '{print $1",porn"}' >diff/$DATE1/rela_task.list
curl --data-binary "@diff/$DATE1/rela_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG" >diff/$DATE1/rela_task_list.list


ls daily_pics/$DATE1/xiaohongshu|awk '{print "https://data.fengkongcloud.com/image/daily_porn/'$DATE1'/xiaohongshu/"$1}' >diff/$DATE1/final_xiaohongshu_url.list
cat phash_porn.list|grep xiaohongshu >diff/$DATE1/final_xiaohongshu_url.list
cat diff/$DATE1/final_xiaohongshu_url.list|awk '{print $1",porn"}' >diff/$DATE1/xiaohongshu_task.list
curl --data-binary "@diff/$DATE1/xiaohongshu_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG" >diff/$DATE1/xiaohongshu_task_list.list


ls daily_pics/$DATE1/xiongmaotv|awk '{print "https://data.fengkongcloud.com/image/daily_porn/'$DATE1'/xiongmaotv/"$1}' >diff/$DATE1/final_xiongmaotv_url.list
cat phash_porn.list|grep xiongmaotv >diff/$DATE1/final_xiongmaotv_url.list
cat diff/$DATE1/final_xiongmaotv_url.list|awk '{print $1",porn"}' >diff/$DATE1/xiongmaotv_task.list
curl --data-binary "@diff/$DATE1/xiongmaotv_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG" >diff/$DATE1/xiongmaotv_task_list.list


ls daily_pics/$DATE1/xiaoying|awk '{print "https://data.fengkongcloud.com/image/daily_porn/'$DATE1'/xiaoying/"$1}' >diff/$DATE1/final_xiaoying_url.list
cat phash_porn.list|grep xiaoying >diff/$DATE1/final_xiaoying_url.list
cat diff/$DATE1/final_xiaoying_url.list|awk '{print $1",porn"}' >diff/$DATE1/xiaoying_task.list
curl --data-binary "@diff/$DATE1/xiaoying_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG" >diff/$DATE1/xiaoying_task_list.list


ls daily_pics/$DATE1/uki|awk '{print "https://data.fengkongcloud.com/image/daily_porn/'$DATE1'/uki/"$1}' >diff/$DATE1/final_uki_url.list
cat phash_porn.list|grep uki >diff/$DATE1/final_uki_url.list
cat diff/$DATE1/final_uki_url.list|awk '{print $1",porn"}' >diff/$DATE1/uki_task.list
curl --data-binary "@diff/$DATE1/uki_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG" >diff/$DATE1/uki_task_list.list


ls daily_pics/$DATE1/yizhibo|awk '{print "https://data.fengkongcloud.com/image/daily_porn/'$DATE1'/yizhibo/"$1}' >diff/$DATE1/final_yizhibo_url.list
cat phash_porn.list|grep yizhibo >diff/$DATE1/final_yizhibo_url.list
cat diff/$DATE1/final_yizhibo_url.list|awk '{print $1",porn"}' >diff/$DATE1/yizhibo_task.list
curl --data-binary "@diff/$DATE1/yizhibo_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG" >diff/$DATE1/yizhibo_task_list.list


ls daily_pics/$DATE1/huanqiubushou|awk '{print "https://data.fengkongcloud.com/image/daily_porn/'$DATE1'/huanqiubushou/"$1}' >diff/$DATE1/final_huanqiubushou_url.list
cat phash_porn.list|grep huanqiubushou >diff/$DATE1/final_huanqiubushou_url.list
cat diff/$DATE1/final_huanqiubushou_url.list|awk '{print $1",porn"}' >diff/$DATE1/huanqiubushou_task.list
curl --data-binary "@diff/$DATE1/huanqiubushou_task.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG" >diff/$DATE1/huanqiubushou_task_list.list


python ../gen_mail.py diff/$DATE1 porn> ${mail_output}

wait
