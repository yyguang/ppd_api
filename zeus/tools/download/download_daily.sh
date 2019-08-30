RANDOM_ID=$RANDOM
SERVER=js    #server可以配置为腾讯或金山，对应tt和js
RENAME_MD5=1 #是否根据md5进行重命名，当进行数据修正时，需要保留原始文件名信息
python get_url_from_db.py $1 $SERVER|grep http >/tmp/tmp_$RANDOM_ID'_url.list'
echo total,`cat /tmp/tmp_$RANDOM_ID'_url.list'|wc -l`

mkdir $2
labels=`cat /tmp/tmp_$RANDOM_ID'_url.list'|awk '{print $2}'|sort|uniq`
for i in $labels
do
  mkdir $2/$i
  cat /tmp/tmp_$RANDOM_ID'_url.list'|awk '{if($2=="'$i'")print $1}' >/tmp/tmp_$RANDOM_ID'_'$i'.list'
  echo $i,`cat /tmp/tmp_$RANDOM_ID'_url.list'|awk '{if($2=="'$i'")print $0}'|wc -l`
  python download_insert.py /tmp/tmp_$RANDOM_ID'_'$i'.list' $2/$i $SERVER $RENAME_MD5
  #echo $i
  #cat /tmp/tmp_$RANDOM_ID'_'$i'.list'|head -n1
done
rm /tmp/tmp_$RANDOM_ID*
