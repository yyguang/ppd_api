#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <teng.cao@foxmail.com> 2018-02-28 15:30:12
import sys
import os
import numpy as np

# 输入图谱结果，输出每个类别的召回率，准确率
# 保存为
if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("usage : tupu_result_file inference_output.txt")
    sys.exit(-1)
  
  label_2_idx = {'normal': 0, 'porn': 1, 'sexy': 2}
  count_invalid_line = 0
  count_file_notexist = 0
  count_no_label = 0
  images = []
  ytrue = []
  ypred = []
  yprob = []
  with open(sys.argv[1], "r") as f:
    for line in f:
      arr = line.strip().split()
      if len(arr) != 4:
        count_invalid_line += 1
        continue
      filename = arr[0]
      pred_label = arr[2]
      ground_label = arr[3]
      if not os.path.exists(filename):
        count_file_notexist += 1
        continue
      if pred_label == 'error':
        count_no_label += 1
        continue
      images.append(filename)
      ytrue.append(ground_label)
      ypred.append(pred_label)
      yprob.append(float(arr[1]))
  print('count_invalid_line: %d' % count_invalid_line)
  print('count_file_notexist: %d' % count_file_notexist)
  print('count_no_label: %d' % count_no_label)
  print('count_valid_files: %d' % len(images))
  
  from sklearn.metrics import precision_recall_fscore_support
  ytrue_np = np.asarray(ytrue)
  ypred_np = np.asarray(ypred)
  labels = label_2_idx.keys()
  labels.sort()
  result = precision_recall_fscore_support(ytrue_np, ypred_np, labels=labels)
  print('labels: %s' % labels)
  print('result: precision recall fbeta_score support')
  print(result)

  with open(sys.argv[2], "w") as f:
    f.write('%d\n' % len(labels))
    for i, item in enumerate(labels):
      f.write('%d %s\n' % (i, item))
    for i, image in enumerate(images):
      ypred_idx = label_2_idx[ypred[i]]
      ytrue_idx = label_2_idx[ytrue[i]]
      probs = [-1] * len(labels)
      probs[ypred_idx] = yprob[i]
      f.write('%s %d %f %f %f %d\n' % (image, ypred_idx, probs[0], probs[1], probs[2], ytrue_idx))
