#encoding:utf-8
import random
import sys
import os

if __name__ == '__main__':
  file_list = sys.argv[1]
  sample_num = int(sys.argv[2])
  all_list = []
  sample_list = []
  with open(file_list,'r')as f:
    for line in f:
      line = line.strip()
      all_list.append(line)
  if sample_num >= len(all_list):
    sample_list = all_list
  else:
    sample_list = random.sample(all_list,sample_num)
  for i in sample_list:
    print(i)

