#encoding:utf-8
import sys
import os
import random
import math
import shutil

_RANDOM_SEED = 0
if __name__ == "__main__":
  tmp = {}
  same_count = {}
  # 读取标签
  f = open(sys.argv[1], "r")
  n_label = int(f.readline().strip())
  labels = [''] * n_label
  for i in range(n_label):
    arr = f.readline().strip().split() 
    label_idx = int(arr[0])
    label_name = arr[1]
    tmp[label_idx] = label_name
  lines = []
  for line in f:
    lines.append(line)
  lines.sort()
  rst = {}
  for line in lines:
    arr = line.strip().split()
    if len(arr) <= 1:
      continue
    url = arr[0]
    if "http" not in url:
      print(url)
      continue
    pred_lb = tmp[int(arr[1])]
    pred1 = tmp[int(arr[-1])]
    arr_ = map(eval, arr[2:-1])
    pred_ = max(arr_)
    if pred1 != pred_lb:
      #print url,pred1,pred_lb,pred_
      type_ = pred1+'&'+pred_lb
      if type_ in rst:
        rst[type_].append(url)
      else:
        rst[type_] = [url]
    else:
      if pred1 not in same_count:
        same_count[pred1] = 1
      else:
        same_count[pred1] += 1
  values =  rst.values()
  sum_ = sum([len(v) for v in values])
  print('diff count: {}'.format(sum_))
  with open('storm_diff_url.txt', 'w') as f:
    for k, v in rst.items():
      random.seed(_RANDOM_SEED)
      split_ratio = float(len(v)) / sum_
      select = random.sample(v, min(int(math.ceil(int(sys.argv[2])*split_ratio)), len(v)))
      print(k, len(v), len(select))
      for item in select:
        try:
          f.write("{}\t{}\n".format(item, k))
        except Exception as e:
          continue
  print('count image nums with same label.')
  for pred in sorted(same_count.keys()):
    print('{} {}'.format(pred, same_count[pred]))

