#encoding:utf-8
import sys
import os


with open(sys.argv[1],'r')as f:
  for line_ in f:
    line = line_.strip()
    filename = os.path.basename(line)
    if "_" in filename:
      label = "not_politics"
    else:
      label = "politics_related"
    print("{},{}".format(line,label))
    

