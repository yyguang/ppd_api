#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <teng.cao@foxmail.com> 2018-01-24 13:52:45
import os
import sys
import shutil
import datetime

# 打标misclassify数据
# 第一个参数为misclassify_analysis.py输出的根目录
# 第二个参数为每个子目录需要打的条数，用逗号隔开，顺序为
# normal_rank_by_porn, normal_rank_by_sexy, porn_rank_by_normal, 
# porn_rank_by_sexy, sexy_rank_by_normal, sexy_rank_by_porn
# 这里以原标注值作为默认标签
# 调用示例：python misclassify_upload.py misclassify_9847_test 200,500,200,200,300,300
if __name__ == "__main__":
  data_dir = sys.argv[1]
  count_subdir_ = sys.argv[2].split(',')
  count_subdir = []
  for count in count_subdir_:
    count_subdir.append(int(count))
  
  PER_TASK_COUNT = 300
  SUB_DIRS = ['normal_rank_by_porn', 'normal_rank_by_sexy',
    'porn_rank_by_normal', 'porn_rank_by_sexy', 'sexy_rank_by_normal',
    'sexy_rank_by_porn']
  
  ts = datetime.datetime.now().strftime('%H%M%S')
  base_dir = "porn_misclassify_updoad_" + ts
  dst_dir = os.path.join(data_dir, base_dir)
  os.makedirs(dst_dir)

  # 构建上传目录，拷贝图片
  total_count = 0
  for i,subdir in enumerate(SUB_DIRS):
    original_cat = subdir.split('_')[0]
    dst_dir_curt = os.path.join(dst_dir, original_cat)
    if not os.path.exists(dst_dir_curt):
      os.makedirs(dst_dir_curt)
    subdir = os.path.join(data_dir, subdir)
    files = os.listdir(subdir)
    count = count_subdir[i]
    assert count <= len(files)
    files = sorted(files)[0:count]
    print('process subdir %s for %d files' % (subdir, count))
    for j,f in enumerate(files):
      prefix = "%06d" % j
      assert f.startswith(prefix)
      src_fn = os.path.join(subdir, f)
      dst_fn = os.path.join(dst_dir_curt, f)
      assert not os.path.exists(dst_fn)
      #print('%s -> %s' % (src_fn, dst_fn))
      shutil.copy(src_fn, dst_fn)
    total_count += count

  # 上传数据
  os.chdir(data_dir) # 必须先进入数据目录所在的同级目录
  cmd = 'python ~/image-tools/label_tools/upload_label_task.py %s \
  category %d' % (base_dir, PER_TASK_COUNT)
  os.system(cmd)
  print('%d total files processed' % total_count)
  #shutil.rmtree(tmp_dir)
