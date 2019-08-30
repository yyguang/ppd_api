# encoding:utf-8
import requests
import json
import os
import argparse
import sys
import time

def add_phash(organization,accessKey,list_id,imgUrl):
  try:
    url = 'https://www.fengkongcloud.com/ApiV2/ListService/lists/'+str(list_id)+'/contents?accessKey='+str(accessKey)
    data = {'organization':str(organization),'serviceId':'POST_IMG','operation':'image_hash','checkItems':["img"],'imgUrl':imgUrl}
    response = requests.post(url,data=json.dumps(data))
    print("添加成功%s" %(imgUrl))
  except Exception as e:
    print e

def upload_imgs(img_dir,image_url_list):
  try:
    os.system("scp -pr {} imgupload@admin1.bj.sm:/mnt/imgupload".format(img_dir))
  except Exception as e:
    print e
  img_list = os.listdir(img_dir)
  with open(image_url_list,'w') as f:
    for path in img_list:
      imgUrl = "https://data.fengkongcloud.com/image/{}/{}".format(img_dir, path)
      f.write(imgUrl+'\n')
      f.flush()
      print imgUrl

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--img_dir',type=str,default='./img_dir',
    help='path to add phash list image dir.'
  )
  parser.add_argument(
    '--organization',type=str,
    help='phash list belongs to which organazation'
  )
  parser.add_argument(
    '--accessKey',type=str,
    help='accessKey'
  )
  parser.add_argument(
    '--list_id',type=str,
    help='phash list id'
  )
  parser.add_argument(
    '--saved_list',type=str,default='./image_url.list',
    help='path to save gen image url'
  )
  parser.add_argument(
    '--sleep',type=int,default=2,
    help='add phash sleep time'
  )
  FLAGS ,unparsed = parser.parse_known_args()
  print(FLAGS)
  if len(unparsed) > 0:
    print("there are unknow args %s " % ','.join(unparsed))
    sys.exit(-1)

  img_dir = FLAGS.img_dir
  organization = FLAGS.organization
  accessKey = FLAGS.accessKey
  list_id = FLAGS.list_id
  image_url_list = FLAGS.saved_list
  sleep_time = FLAGS.sleep

  upload_imgs(img_dir,image_url_list)
  print("生成url saved at %s" %(image_url_list))
  with open(image_url_list,'r')as f:
    for url in f:
      add_phash(organization,accessKey,list_id,url)
      time.sleep(sleep_time)
