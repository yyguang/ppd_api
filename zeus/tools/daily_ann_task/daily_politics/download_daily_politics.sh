DATE1=$(date -d "1 days ago" +%Y%m%d)
DATE2=$(date +%Y%m%d)
DATE3=$(date -d "14 days ago" +%Y%m%d)
ad_dir=$(cd `dirname $0`; pwd)
cd $ad_dir
mkdir -p daily_pics/$DATE1

mkdir -p daily_pics/$DATE1/politics

mkdir daily_faces/

python ../download_insert.py  diff/$DATE1/politics.list daily_pics/$DATE1/politics politics >diff/$DATE1/download_politics.log 2>&1
wait

#上传gpu进行人脸剪裁
scp -i ishumei-id_rsa -r daily_pics/$DATE1 fujunhao@gpu.bj-js.sm:/home/fujunhao/daily_politics/be-image-detector/test/daily-pics/
wait
#通过ssh 远程执行人脸剪裁操作
ssh -i ishumei-id_rsa fujunhao@gpu.bj-js.sm "sh /home/fujunhao/daily_politics/be-image-detector/test/face_detect_task.sh"
wait
#取回剪裁好的人脸
scp -i ishumei-id_rsa -r fujunhao@gpu.bj-js.sm:/home/fujunhao/daily_politics/be-image-detector/test/daily_faces/$DATE1/ daily_faces

#上传upload服务
rsync -av --delete daily_faces/$DATE1 imgupload@admin1.bj.sm:/mnt/imgupload/daily_politics/

#生成标注label
ls daily_faces/$DATE1/|head -n100|awk '{print "https://data.fengkongcloud.com/image/daily_politics/'$DATE1'/"$1}' >diff/$DATE1/final_shumei_politics_url.list
python gen_label.py diff/$DATE1/final_shumei_politics_url.list >diff/$DATE1/shumei_politics_task.list
sort -r diff/$DATE1/shumei_politics_task.list >diff/$DATE1/shumei_politics_task-new.list
curl --data-binary "@diff/$DATE1/shumei_politics_task-new.list" -XPOST "https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?organization=RlokQwRlVjUrTUlkIqOg&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG&taskType=category-politics" >diff/$DATE1/shumei_politics_task_list.list
wait
python ../gen_mail.py diff/$DATE1 politics > $1
wait
