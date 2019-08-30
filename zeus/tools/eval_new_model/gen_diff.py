#encoding:utf-8
import sys
import os
import random
import math
import shutil

_RANDOM_SEED = 0
if __name__ == "__main__":
  tmp = {}
  # 读取标签
  f = open(sys.argv[1], "r")
  n_label = int(f.readline().strip())
  labels = [''] * n_label
  for i in range(n_label):
    arr = f.readline().strip().split() 
    label_idx = int(arr[0])
    label_name = arr[1]
    tmp[label_idx] = label_name
  lines= []
  for line in f:
    lines.append(line)
  lines.sort()
  rst = {}
  for line in lines:
    arr = line.strip().split()
    if len(arr) <= 1:
      continue
    fullpath = arr[0]
    pred_lb = tmp[int(arr[1])]
    pred1 = fullpath.strip().split('_')[-2]
    arr_ = map(eval, arr[2:-1])
    pred_ = max(arr_)
    if pred1 != pred_lb:
      print fullpath,pred1,pred_lb,pred_
      basename = fullpath.split('.jpg')[0]
      name = basename+'&'+pred_lb+'&'+str(pred_)+'.jpg'
      type_ = pred1+'&'+pred_lb
      new_name = os.path.basename(name)
      if rst.has_key(type_):
        rst[type_].append([os.path.join(type_,new_name),fullpath])
      else:
        rst[type_] = [os.path.join(type_,new_name),fullpath]
  values =  rst.values()
  sum_ = sum([len(v) for v in values])
  for k,v in rst.items():
    random.seed(_RANDOM_SEED)
    split_ratio = float(len(v)) / sum_
    select = random.sample(v, int(math.ceil(int(sys.argv[3])*split_ratio)))
    print k, len(v), len(select)
    for item in select:
      try:
        src = os.path.join(item[1])
        new_dir = os.path.join(sys.argv[2], item[0])
        dir_ = os.path.abspath(os.path.dirname(new_dir))
        if not os.path.exists(dir_):
          os.makedirs(dir_)
        shutil.copyfile(src,new_dir)
      except Exception as e:
       continue
