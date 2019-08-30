#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author: chenyunkuo2018-04-09 15:57:16

# 输入各家预测结果文件（inference_output.txt）
# 用于检索数美、图谱、百度、阿里中，任意两家结果不同的数据
# 最终生成的目录结果标签使用数美模型的预测结果

import os
import shutil
import argparse

# 读取inference_output文件
def read_inference_output(result_file):
  if not result_file:
    return {}, {}, {}
  rows = open(result_file, "r").readlines()
  labels_num = int(rows[0])
  labels = {}
  md5_fullpath = {}
  md5_pred = {}
  for r in rows[1: labels_num + 1]:
    idx, label = r.strip().split()
    labels[idx] = label
  for r in rows[labels_num + 1:]:
    img, pred = r.strip().split()[:2]
    md5 = os.path.basename(img)
    md5_fullpath[md5] = img
    md5_pred[md5] = pred
  return md5_pred, md5_fullpath, labels

# 得到各家结果
def get_all_result(img, shumei, tupu, baidu, ali):
  return "\t".join([shumei.get(img, ""), tupu.get(img, ""), baidu.get(img, ""), ali.get(img, "")])

# 检索diff数据
def catch_diff(shumei, tupu, baidu, ali):
  diff_images = {}
  for img, v in shumei.items():
    if img in tupu and tupu[img] != v:
      diff_images[img] = get_all_result(img, shumei, tupu, baidu, ali)
      continue
    elif img in baidu and baidu[img] != v:
      diff_images[img] = get_all_result(img, shumei, tupu, baidu, ali)
      continue
    elif img in ali and ali[img] != v:
      diff_images[img] = get_all_result(img, shumei, tupu, baidu, ali)
  return diff_images


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--shumei_result', type=str, default='inference_output_shumei.txt',
    help='调用 evaluation.py 生成的数美结果文件.'
  )
  parser.add_argument(
    '--tupu_result', type=str, default='',
    help='调用图谱api生成的结果文件.'
  )
  parser.add_argument(
    '--baidu_result', type=str, default='',
    help='调用百度api生成的结果文件.'
  )
  parser.add_argument(
    '--ali_result', type=str, default='',
    help='调用阿里api生成的结果文件.'
  )
  parser.add_argument(
    '--is_output', type=int, default=0,
    help='是否将diff数据保存到output目录.'
  )
  parser.add_argument(
    '--output', type=str, default='diff_to_upload',
    help='diff数据保存目录，仅当is_output为1时有效.'
  )
  parser.add_argument(
    '--diff_images_file', type=str, default='diff_images.txt',
    help='diff数据信息文件，格式为"img_md5\tshumei_label\tother_label...".'
  )
  FLAGS, unparsed = parser.parse_known_args()
  print(FLAGS)
  if len(unparsed) > 0:
    print("there are unknow args %s " % ','.join(unparsed))
    sys.exit(-1)
  
  shumei_result = FLAGS.shumei_result
  tupu_result = FLAGS.tupu_result
  baidu_result = FLAGS.baidu_result
  ali_result = FLAGS.ali_result
  is_output = FLAGS.is_output
  output = FLAGS.output
  diff_images_file = FLAGS.diff_images_file
 
  shumei_pred, md5_fullpath, labels = read_inference_output(shumei_result)
  tupu_pred, _, __ = read_inference_output(tupu_result)
  baidu_pred, _, __ = read_inference_output(baidu_result)
  ali_pred, _, __ = read_inference_output(ali_result)
  diff_images = catch_diff(shumei_pred, tupu_pred, baidu_pred, ali_pred)

  # 生成diff数据信息文件
  with open(diff_images_file, "w") as f:
    f.write("{}\n".format(len(labels)))
    for i in range(len(labels)):
      f.write("{} {}\n".format(i, labels[str(i)]))
    f.write("image\tshumei\ttupu\tbaidu\tali\n")
    for d, v in diff_images.items():
      f.write("{}\t{}\n".format(md5_fullpath[d], v))

  if is_output:
    if os.path.isdir(output):
      shutil.rmtree(output)
    for l in labels.values():
      os.system("mkdir -p {}".format(os.path.join(output, l)))
    # 将diff图片转移到output下，子目录使用shumei预测结果
    for d in diff_images:
      os.system("cp {} {}".format(md5_fullpath[d], os.path.join(output, labels[shumei_pred[d]])))
