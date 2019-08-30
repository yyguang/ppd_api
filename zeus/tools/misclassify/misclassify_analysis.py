#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <teng.cao@foxmail.com> 2018-01-22 11:40:34
import os
import sys
import shutil
import argparse
from collections import defaultdict

def process_mode0(labels, result, is_argmax):
  nlabel = len(labels)
  for i, label in enumerate(labels):
    dir_label = os.path.join(output, label)
    result_label = result[i]
    for j in range(nlabel):
      if j != i:
        sorted_result = sorted(result_label, key=lambda s: float(s[1 + j]), reverse=True)
        dir_output = "%s_rank_by_%s" % (dir_label, labels[j])
        os.makedirs(dir_output)
        count_output = min(len(sorted_result), class_topk[i])
        for k in range(count_output):
          item = sorted_result[k]
          rate_list = [float(it) for it in item[1:]]
          if is_argmax and rate_list.index(max(rate_list)) != j:
            continue
          src_fn = item[0]
          basename = "|".join(src_fn.split('/')[-3:])
          dst_fn = "%s/%06d|%.4f|%s" % (dir_output, k, float(item[1 + j]), basename)
          assert not os.path.exists(dst_fn)
          shutil.copy(src_fn, dst_fn)


def process_mode1(labels, result, is_argmax):
  nlabel = len(labels)
  for i, label in enumerate(labels):
    dir_label = os.path.join(output, label)
    result_label = result[i]
    j = i
    sorted_result = sorted(result_label, key=lambda s: float(s[1 + j]), reverse=False)
    dir_output = "%s_rank_by_%s" % (dir_label, labels[j])
    os.makedirs(dir_output)
    count_output = min(len(sorted_result), class_topk[i])
    for k in range(count_output):
      item = sorted_result[k]
      rate_list = [float(it) for it in item[1:]]
      if is_argmax and rate_list.index(max(rate_list)) == j:
        continue
      src_fn = item[0]
      basename = "|".join(src_fn.split('/')[-3:])
      dst_fn = "%s/%06d|%.4f|%s" % (dir_output, k, float(item[1 + j]), basename)
      assert not os.path.exists(dst_fn)
      shutil.copy(src_fn, dst_fn)


# 对inference_output.txt进行分析，生成误分类的数据
# 模式0：标注为a，识别为b，对于n分类，将生成总共n*(n-1)个目录
# 模式1：标注为a，按照识别为a的概率由小到大排序，将生成总共n个目录
# --inference_output参数为inference_output.txt
# --output参数为输出的根目录
# --topk参数为每个目录最多包含多少条样本
# --mode参数为模式，支持[0, 1]
# --is_argmax参数表示是否只筛选argmax误分类结果
# 调用示例：python misclassify_analysis.py --inference_output inference_output_9847_test.txt --output misclassify_9847_test --topk 1000 --mode 0
# 输出根目录下会包括六个子目录，比如normal_rank_by_porn表示标注为normal的样本按照porn的概率由高到低排序
# 文件名示例：000112|0.9906|daily_20171114_test|sexy|8ec44189b6e84b9c61c0f5d455b9d338.jpg
# 由逗号分隔：序号，相应类别的概率，文件来源目录，真实类别，md5
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--inference_output', type=str, default='inference_output.txt',
    help='The path of inference output file.'
  )
  parser.add_argument(
    '--output', type=str, default='misclassify_output',
    help='The output path to save misclassification cases.'
  )
  parser.add_argument(
    '--topk', type=int, default=1000,
    help='The max num of cases on every misclassification.'
  )
  parser.add_argument(
    '--mode', type=int, default=0,
    help='These are 2 modes, {0, 1}.'
  )
  parser.add_argument(
    '--is_argmax', type=int, default=1,
    help='Whether only catch misclassification cases by argmax.'
  )
  parser.add_argument(
    '--proportion', type=int, default=0,
    help='Whether to keep the proportion among classes, and the count of cases is topk * class num if opened.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  print(FLAGS)
  if len(unparsed) > 0:
    print("there are unknow args %s " % ','.join(unparsed))
    sys.exit(-1)

  inference_output = FLAGS.inference_output
  output = FLAGS.output
  topk = FLAGS.topk
  mode = FLAGS.mode
  is_argmax = FLAGS.is_argmax
  proportion = FLAGS.proportion
  assert mode in {0, 1}
  assert is_argmax in {0, 1}
  labels = []
  result = []
  pred_dict = defaultdict(lambda: [])
  with open(inference_output, "r") as f:
    nlabel = int(f.readline().strip())
    for i in range(nlabel):
      parts = f.readline().strip().split()
      assert len(parts) == 2
      assert int(parts[0]) == i
      labels.append(parts[1])
      result.append([])
    for line in f:
      parts = line.strip().split()
      assert len(parts) == nlabel + 3
      true_label = int(parts[-1])
      pred_dict[true_label].append(int(parts[1]))
      del parts[1]
      result[true_label].append(parts[0: -1])
 
  # 打印正确召回数:
  for idx, label in enumerate(labels):
    right_count = 0
    for pred in pred_dict[idx]:
      if pred == idx:
        right_count += 1
    print("{} right recall: {}".format(label, right_count))
 
  # 计算各类别比例
  class_topk = defaultdict(lambda: topk) 
  if proportion:
    total_count = 0
    for i in range(nlabel):
      total_count += len(result[i])
    for i in range(nlabel):
      class_topk[i] = int(nlabel * topk * len(result[i])/ total_count)
  print 'class topk by proportion: ', dict(class_topk)

  # 输出结果
  print('labels: %s' % labels)
  if os.path.exists(output):
    shutil.rmtree(output)
  os.makedirs(output)
  if mode == 0:
    process_mode0(labels, result, is_argmax)
  elif mode == 1:
    process_mode1(labels, result, is_argmax)
  
