#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author: chenyunkuo2018-08-20 17:47:08
# 通过传入类别映射list将多分类的inference映射成更少分类的inference
# Usage: python inference_classes_num_reduction.py inference_output.txt classes_map.list

import sys
import argparse

FLAGS = None

# 得到类别归并的dict及新的标签list
def get_classes_map_dict(classes_map_file, labels):
  classes_map_dict = {}
  for row in open(classes_map_file, 'r'):
    ori_label, new_label = row.strip().split()
    classes_map_dict[labels.index(ori_label)] = new_label
  new_labels = sorted(set(classes_map_dict.values()))
  return classes_map_dict, new_labels

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--inference', type=str, default='inference_output.txt',
    help='The path of inference output file.'
  )
  parser.add_argument(
    '--output', type=str, default='inference_less_classes.txt',
    help='The output path to save new inference.'
  )
  parser.add_argument(
    '--classes_map', type=str, default="classes_map.list",
    help='The list file saved more classes mapping to less classes.'
  )
  parser.add_argument(
    '--keep_pred', type=int, default=1,
    help='Whether to keep old pred label, else save new argmax label on less classes.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  print(FLAGS)
  if len(unparsed) > 0:
    print("there are unknow args %s " % ','.join(unparsed))
    sys.exit(-1)
  inference = FLAGS.inference
  output = FLAGS.output
  classes_map_file = FLAGS.classes_map
  keep_pred = FLAGS.keep_pred
 
  labels = []
  recall_imgs = {}
  for row in open(inference, 'r'):
    if len(row.strip().split()) == 2:
      idx, label = row.strip().split()
      labels.append(label)
  
  classes_map_dict, labels = get_classes_map_dict(classes_map_file, labels)
  
  with open(output, "w") as f:
    f.write("{}\n".format(len(labels)))
    for i in range(len(labels)):
      f.write("{} {}\n".format(i, labels[i]))
    for row in open(inference, 'r'):
      if len(row.strip().split()) > 2:
        img = row.strip().split()[0]
        pred_lbl = int(row.strip().split()[1])
        pred_lbl = labels.index(classes_map_dict[pred_lbl])
        ground_truth = int(row.strip().split()[-1])
        ground_truth = labels.index(classes_map_dict[ground_truth])
        y_all = row.strip().split()[2: -1]
        y_all_new = [0.] * len(labels)
        for i in range(len(y_all)):
          prob = float(y_all[i])
          y_all_new[labels.index(classes_map_dict[i])] += prob
        # 取新的较少类别的概率argamx为预测标签
        if not keep_pred:
          pred_lbl = y_all_new.index(max(y_all_new))
        f.write("{}\t{}\t".format(img, pred_lbl))
        f.write("\t".join([str(prob) for prob in y_all_new]))
        f.write("\t{}\n".format(ground_truth))
