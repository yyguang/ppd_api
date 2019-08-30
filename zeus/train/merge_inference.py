#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author: chenyunkuo2018-09-11 14:27:21
# 可以合并多个inference文件，回归到三分类，评估ensemble效果.
# 目前仅适用于porn模型,最多回归到4分类

import os
import sys
import argparse
from collections import defaultdict

FLAGS = None
label_level = {'normal': 0, 'sexy': 1, 'porn': 2, 'exposed': 3}

# 读取多个inference文件，根据ensemble方法，返回一个3分类的result
def read_inference(inferences, esb_method, classes_map_dict):
  img_gt = {}
  label_indices = {}
  idx_label = {}
  label_prob = {}
  indices = 0
  for label in sorted(set(classes_map_dict.values())):
    label_prob[label] = defaultdict(lambda: 0.0)
    label_indices[label] = indices
    idx_label[indices] = label
    indices += 1
  print(label_indices)
  print(idx_label)
  img_pred = defaultdict(lambda: label_indices['normal'])
  result = []
  inf_count = 0
  for inf in inferences:
    inf_count += 1
    labels = {}
    for row in open(inf, 'r'):
      if len(row.strip().split()) == 2:
        idx, label = row.strip().split()
        labels[idx] = label
      elif len(row.strip().split()) > 2:
        img = row.strip().split()[0]
        pred_lbl = row.strip().split()[1]
        probs = [float(i) for i in row.split()[2: -1]]
        for idx, p in enumerate(probs):
          trans_label = classes_map_dict[labels[str(idx)]]
          label_prob[trans_label][img] += p
        pred_label = label_indices[classes_map_dict[labels[pred_lbl]]]
        # argmax 高召回为准
        if label_level[idx_label[pred_label]] > label_level[idx_label[img_pred[img]]]:
          img_pred[img] = pred_label
        img_gt[img] = str(label_indices[classes_map_dict[labels[row.strip().split()[-1]]]])
  for img in img_gt.keys():
    for img_prob_label in label_prob.values():
      img_prob_label[img] = img_prob_label[img]/inf_count
  if esb_method == 'average':
    for img in img_pred.keys():
      rates = []
      for label in sorted(label_indices.keys()):
        rates.append(label_prob[label][img])
      argmax_pred = rates.index(max(rates))
      img_pred[img] = argmax_pred
  result.append(str(len(label_indices)))
  for label in sorted(label_indices.keys()):
    result.append("{} {}".format(label_indices[label], label))
  for img in img_pred.keys():
    line = []
    line.append(img)
    line.append(str(img_pred[img]))
    for label in sorted(label_indices.keys()):
      line.append(str(label_prob[label][img]))
    line.append(str(img_gt[img]))
    result.append("\t".join(line))
  return result


# 读取classes map，返回label归并的dict
def read_classes_map(classes_map):
  assert os.path.exists(classes_map)
  classes_map_dict = {}
  for row in open(classes_map, 'r'):
    assert len(row.strip().split()) == 2
    ori_label, new_label = row.strip().split()
    classes_map_dict[ori_label] = new_label
  print(classes_map_dict)
  return classes_map_dict


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--inference_list', type=str, default='inference_output_3classes.txt,inference_output_4classes.txt',
    help='The list of inference output files, seperated by ",".'
  )
  parser.add_argument(
    '--output', type=str, default='inference_collect.txt',
    help='The output path to save new inference.'
  )
  parser.add_argument(
    '--ensemble_method', type=str, default='argmax',
    help='The method to ensemble multi models, exp: argmax, average.'
  )
  parser.add_argument(
    '--classes_map', type=str, default="",
    help='输入类别映射的list文件，按照该文件进行类别融合及ignore被融合的类别，e.g. "porn_8classes_to_3classes.list".'
  )
  FLAGS, unparsed = parser.parse_known_args()
  print(FLAGS)
  if len(unparsed) > 0:
    print("there are unknow args %s " % ','.join(unparsed))
    sys.exit(-1)
  inferences = FLAGS.inference_list.split(',')
  output = FLAGS.output
  esb_method = FLAGS.ensemble_method
  classes_map = FLAGS.classes_map
  classes_map_dict = read_classes_map(classes_map)
  result = read_inference(inferences, esb_method, classes_map_dict)
  with open(output, 'w') as f:
    f.write("\n".join(result) + "\n")

