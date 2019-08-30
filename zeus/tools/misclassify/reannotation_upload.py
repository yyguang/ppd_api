#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <teng.cao@foxmail.com> 2018-02-02 12:21:45
import os
import sys
import shutil
import datetime
import random

# 标准修改较大时，需要重标注原有数据（或一部分）时，采用该脚本进行重标注数据上传
# 数据生成后需要scp到北京腾讯开发机再上传到服务器
# 第一个参数为数据集的定义文件
# 第二个参数为类别，比如porn或sexy或normal（对应porn）；或者violence的对应类型
# 第三个参数为总量，如果为0则表示所有数据都上传，否则随机选取相应的数量上传
# 第四个参数为数据类型，比如porn或violence
# 对于生成的数据（gen打头）不进行处理
# 这里以原标注值作为默认标签
# 调用示例：python reannotation_upload.py dataset_test.txt porn 25000 porn
if __name__ == "__main__":
  dataset_define = sys.argv[1]
  desired_category = sys.argv[2]
  desired_count = int(sys.argv[3])
  task_type = sys.argv[4]
  data_dir = '/home/caoteng/data/%s-data-correct' % task_type

  dbs = []
  assert os.path.exists(dataset_define)
  with open(dataset_define, 'r') as f:
    for line in f:
      line = line.strip()
      assert os.path.exists(line)
      dbs.append(line)

  # 查找所有文件，存放到all_files中
  all_files = []
  for db in dbs:
    print('process databse %s' % db)
    db_basename = os.path.basename(db)
    if db_basename.startswith("gen"):
      print('starts with gen, ignored')
      continue
    dirs = os.listdir(db)
    for dir in dirs:
      if dir != desired_category:
        continue
      fulldir = os.path.join(db, dir)
      if not os.path.isdir(fulldir):
        continue
      files = os.listdir(fulldir)
      for f in files:
        fullpath = os.path.join(fulldir, f)
        assert fullpath.endswith('.jpg')
        if os.path.isfile(fullpath):
          all_files.append(fullpath)

  # 打印all_files信息
  print('key %s: len: %d' % (desired_category, len(all_files)))
  random.shuffle(all_files)
  if desired_count == 0:
    desired_count = len(all_files)
  desired_count = min(desired_count, len(all_files))
  all_files = all_files[0:desired_count]

  PER_TASK_COUNT = 300
  
  ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  base_dir = "%s_reannotation_updoad_%s_%d_%s" % (task_type, desired_category, desired_count, ts)
  dst_dir = os.path.join(data_dir, base_dir, desired_category)
  os.makedirs(dst_dir)

  # 构建上传目录，拷贝图片
  total_count = 0
  for j, f in enumerate(all_files):
    post_fix = "|".join(f.split('/')[-3:])
    dst_basename = '%06d|0.0000|%s' % (j, post_fix)
    dst_fn = os.path.join(dst_dir, dst_basename)
    print('%s -> %s' % (f, dst_fn))
    assert not os.path.exists(dst_fn)
    shutil.copy(f, dst_fn)
    #000249|0.9369|liaoai_au_20170921|normal|14a3e58f6013ed0d60d929ed467d00df.jpg
    total_count += 1
  print('%d total files copied' % total_count)

  # 上传数据
  #os.chdir(data_dir) # 必须先进入数据目录所在的同级目录
  #cmd = 'python ~/image-tools/label_tools/upload_label_task.py %s \
  #category %d' % (base_dir, PER_TASK_COUNT)
  #os.system(cmd)
  #print('%d total files processed' % total_count)
