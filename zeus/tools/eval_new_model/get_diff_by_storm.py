#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2018/10/19 下午12:03
# @Author  : yunkchen
# @File    : get_diff_by_storm.py
# Usage    : python get_diff_by_storm.py storm.log online.log porn_3 labels.txt 1

import re
import sys

model_names = {
  "porn_3": "porn_recognition",
  "porn_4": "4classes_porn_recognition",
  "porn_liebao": "liebao_porn_recognition",
  "terror": "terror_recognition",
  "general_hwad": "general_ad_hwad_recognition",
  "taobao_hwad": "taobao_hwad_recognition",
  "xiaohongshu_hwad": "xiaohongshu_hwad_recognition",
  "politics": "politics_recognition"
}

def read_labels(labels_file):
  labels = {}
  idx = 0
  for lbl in open(labels_file, 'r'):
    lbl = lbl.strip()
    labels[idx] = lbl
    labels[lbl] = idx
    idx += 1
  return labels

def read_log(log, labels, model, is_tactics):
  img_pred = {}
  for row in open(log, 'r'):
    if 'detail' in row and model in row:
      url, label, result = get_detail(row.strip(), model)
      if url and label:
        img_pred[url] = labels[label]
        if is_tactics:
          if model == 'porn_recognition':
            label = porn3_on_tactics([result[labels[0]], result[labels[1]], result[labels[2]]])
            img_pred[url] = label
          elif model == '4classes_porn_recognition':
            label = porn4_on_tactics([result[labels[0]], result[labels[1]], result[labels[2]], result[labels[3]]])
            img_pred[url] = label
  return img_pred

def get_detail(line, model):
  url_pat = re.compile('[a-zA-z]+://[^\s]*')
  detail_pat = re.compile('detail=(.*)\s')
  try:
    url = re.findall(url_pat, line)[0]
    detail = eval(re.findall(detail_pat, line)[0].replace('true', 'True').replace('false', 'False').replace('null', 'None'))
    label = detail[model]['label']
    result = detail[model]['result']
    return url, label, result
  except Exception as e:
    #print(e, line)
    return None, None, None

# 线上色情三分类策略
# 3分类标签顺序:normal|porn|sexy
# 3分类线上策略:
#  色情:porn > 0.8 or (normal < 0.2 and porn > sexy)
#  性感:normal < 0.2 and sexy > porn
#  正常:不满足上述两条中任一规则
def porn3_on_tactics(probs):
  assert len(probs) == 3
  if probs[1] > 0.8 or (probs[0] < 0.2 and probs[1] > probs[2]):
    return 1
  elif probs[0] < 0.2 and probs[2] > probs[1]:
    return 2
  else:
    return 0

# 线上色情四分类策略
# 4分类线上策略:
#  重色:serious_porn > 0.8 否则将重色概率加入到porn，然后按照3分类策略进行判别.
def porn4_on_tactics(probs):
  if probs[0] > 0.8:
    return 0
  else:
    probs[2] += probs[0]
    return porn3_on_tactics(probs[1:]) + 1


if __name__ == '__main__':
  storm_log = sys.argv[1]
  online_log = sys.argv[2]
  model = model_names[sys.argv[3]]
  labels = read_labels(sys.argv[4])
  is_tactics = int(sys.argv[5])  # 是否使用线上策略预测标签
  storm_imgs = read_log(storm_log, labels, model, is_tactics)
  online_imgs = read_log(online_log, labels, model, is_tactics)
  with open('inference_ouput.txt', 'w') as f:
    f.write(str(int(len(labels)/2)) + "\n")
    for i in range(int(len(labels)/2)):
      f.write("{} {}\n".format(i, labels[i]))
    probs = "0.\t" * int(len(labels)/2)
    for img in sorted(list(set(storm_imgs.keys()) & set(online_imgs.keys()))):
      f.write("{url}\t{pred}\t{probs}{gt}\n".format(url=img,
                                                    pred=storm_imgs[img],
                                                    probs=probs,
                                                    gt=online_imgs[img]))
