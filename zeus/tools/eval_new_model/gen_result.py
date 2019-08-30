#encoding:utf-8
import sys
import os
import numpy as np

with open(sys.argv[1],'r')as f:
  rst1= {}
  rst2 ={}
  pred1_ = []
  pred2_ = []
  for line_ in f:
    line = line_.strip().split('\t')
    gt = line[-1]
    fullpath = line[0]
    basename = os.path.basename(fullpath)
    pred1 = basename.split('_')[1]
    pred2 = basename.split('&')[1]
    if gt in rst1:
      if gt == pred1:
        rst1[gt].append(1)
      else:
        rst1[gt].append(0)
    else:
      if gt == pred1:
        rst1[gt] = [1]
      else:
        rst1[gt] = [0] 
    if gt in rst2:
      if gt == pred2:
        rst2[gt].append(1)
      else:
        rst2[gt].append(0)
    else:
      if gt == pred2:
        rst2[gt] = [1]
      else:
        rst2[gt] = [0]
sum1 =0
sum2 =0
print("类别\t线上模型正确\t最新模型正确\t都错误\t总数")
for k, v in rst1.items():
   sum1+=v.count(1)
   sum2+=rst2[k].count(1)
   print("{}\t{}\t{}\t{}\t{}".format(k,v.count(1),rst2[k].count(1),(len(v)-v.count(1)-rst2[k].count(1)),len(v)))

print("线上模型正确总数{}\t最新模型正确总数{}".format(sum1,sum2))
