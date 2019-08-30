#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author: chenyunkuo 2018-06-01 19:53:18

import random
import sys

text = sys.argv[1]
out_num = int(sys.argv[2])

bak_list = []
shuf_list = []
for row in open(text, 'r'):
  bak_list.append(row)
  shuf_list.append(row)

random.shuffle(shuf_list)
rm_set = set(shuf_list[:len(shuf_list)-out_num])
with open(text, "w") as f:
  for r in bak_list:
    if r not in rm_set:
      f.write(r)
