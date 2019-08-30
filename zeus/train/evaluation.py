#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Shi Xing <shi-xing-xing@163.com> 2017-09-15 23:50:31
#        Tony Tsao <teng.cao@foxmail.com> 2017-2018

from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, \
     precision_recall_curve, average_precision_score, mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
import shutil
import notebook_util
import datetime
import math
from sm_sequence import SMSequence
from scipy import interpolate
import argparse

FLAGS = None

# 返回：真值标签索引，预测为每个类别的概率，预测标签索引，全路径
# 对于图谱数据，只有所推定类型的概率，其他概率值均为-1
def read_file(fn, file_filter=''):
  fullpath = [] # 为了支持保存misclassified的信息，需要保存fullpath
  pred = []     # predict probability
  lbl = []      # ground truth label index
  pred_lbl = [] # prediction label index

  # 读取标签
  f = open(fn, "r")
  n_label = int(f.readline().strip())
  labels = [''] * n_label
  for i in range(n_label):
    arr = f.readline().strip().split() 
    label_idx = int(arr[0])
    label_name = arr[1]
    labels[label_idx] = label_name

  # 读取结果
  lines = []
  for line in f:
    if file_filter != '':
      if line.find(file_filter) != -1:
        lines.append(line)
    else:
      lines.append(line)
  lines.sort()

  for line in lines:
    arr = line.strip().split()
    fullpath.append(arr[0])
    pred_lbl.append(int(arr[1]))
    pred.append(arr[2:-1])
    lbl.append(int(arr[-1]))
  f.close()
  return (lbl, pred, pred_lbl, fullpath, labels)

# save fn and fp according to their scores for our algorithm
def save_misclassified(name, lbl, pred, topk=1000):
  basedir = "misclassified"
  if os.path.exists(basedir):
    shutil.rmtree(basedir)
  os.makedirs(basedir)
  fn = os.path.join(basedir, 'condition-positive-rank-by-normal-prob')
  fp = os.path.join(basedir, 'condition-negative-rank-by-porn-prob')
  os.makedirs(fn)
  os.makedirs(fp)
  fn_list = [] # 按normal的概率降序排列
  fp_list = [] # 按porn的概率降序排列
  for j in range(len(lbl)):
    if lbl[j] == 1:
      fn_list.append((name[j], 1-pred[j]))
    elif lbl[j] == 0:
      fp_list.append((name[j], pred[j]))
    else:
      assert False

  sorted_fn_list = sorted(fn_list, key=lambda x:x[1], reverse=True) 
  sorted_fp_list = sorted(fp_list, key=lambda x:x[1], reverse=True)
  fn_to_process = min(len(sorted_fn_list), topk)
  for i in range(fn_to_process):
    t = sorted_fn_list[i]
    dst_fn = fn + ('/%06d_%.4f.jpg' % (i, t[1]))
    shutil.copy(t[0], dst_fn)
  fp_to_process = min(len(sorted_fp_list), topk)
  for i in range(fp_to_process):
    t = sorted_fp_list[i]
    dst_fn = fp + ('/%06d_%.4f.jpg' % (i, t[1]))
    dst_fn = fp + ('/%06d_%.4f.jpg' % (i, t[1]))
    shutil.copy(t[0], dst_fn)

# usage
#def usage():
#    print('usage#1: %s --model --input inference_output.txt inference_output_baidu.txt' % sys.argv[0])
#    print('label.txt: label definition file output from retrain.py')
#    print('inference_output.txt: inference result of the test samples') 
#    print('  line format: 文件名 预测label的索引 预测为label=0的概率 预测为label=1的概率 真实label的索引')
#    print('inference_output_baidu.txt(optional): the inference output from')
#    print('  baidu API with difference: the label is predefined as 0-normal,1-porn, 2-sexy')
#    print('  line format: 文件名 预测label的索引 预测为label=normal的概率 预测为label=porn的概率 真实label的索引')

# get prediction results (prob for all labels) and true labels 
# label index and label sequence is same as in generator
def calc_pred_and_labels_complete(generator, model, multiprocessing, workers=8):
  from tensorflow.keras.utils import Sequence
  is_sequence = isinstance(generator, Sequence)
  y = generator.classes
  if is_sequence:
    y_pred = model.predict_generator(generator=generator,
        workers=workers, use_multiprocessing=multiprocessing)
  else:
    total_files = len(generator.filenames)
    steps_per_epoch = int(math.ceil(total_files / float(generator.batch_size)))
    y_pred = model.predict(generator, steps=steps_per_epoch)
  assert len(y_pred) == len(y)
  return (y, y_pred)

def get_pred_label(y_preds):
  pred_label = []
  for y_pred in y_preds:
    y_all = y_pred.tolist()
    max_idx = y_all.index(max(y_all)) # by argmax
    pred_label.append(max_idx)
  return pred_label

# 输入calc_pred_and_labels_complete的结果，评估每个类别的ap和auc
def calc_ap_and_auc(y, y_pred):
  assert len(y_pred) > 0
  nlabel = len(y_pred[0].tolist())
  
  ap_all = []
  auc_all = []
  for i in range(nlabel):
    y_true = []
    y_score = []
    for label_idx in y:
      if label_idx == i:
        y_true.append(1)
      else:
        y_true.append(0)
    for pred in y_pred:
      y_all = pred.tolist()
      y_score.append(y_all[i])
    y_true_np = np.asarray(y_true)
    y_score_np = np.asarray(y_score)
    ap = average_precision_score(y_true_np, y_score_np)
    auc = roc_auc_score(y_true_np, y_score_np)
    ap_all.append(ap)
    auc_all.append(auc)
  return ap_all, auc_all

# 输入calc_pred_and_labels_complete的结果，评估每个类别的precision和recall, F1
# 允许输入classes_map的dict，实现类别映射
def calc_precision_recall_f1(y, y_pred, class_indices_map={}):
  assert len(y_pred) > 0
  nlabel = len(y_pred[0].tolist())
  labels = range(nlabel)
  y_pred2 = get_pred_label(y_pred)
  if len(class_indices_map) > 0:
    y_new = []
    y_pred_new = []
    assert len(y) == len(y_pred)
    for i in range(len(y)):
      y_new.append(class_indices_map[y[i]])
      y_pred_new.append(class_indices_map[y_pred2[i]])
    y = y_new
    y_pred2 = y_pred_new
    labels = range(len(set(class_indices_map.values())))
  result = precision_recall_fscore_support(y, y_pred2, labels=labels)
  return result[0].tolist(), result[1].tolist(), result[2].tolist()

def eval_model(model_file, label_file, image_list, output, 
    multiprocessing, image_size, interpolation='bilinear', letter_box=0, batch_size=256, special='', classes_map='', keep_pred=1, optimizing_type='classify'):
  if not os.path.exists(image_list):
    print('image_list not exist: %s' % image_list)
    sys.exit(-1)
  if not os.path.exists(model_file):
    print("model not exist: %s" % model_file)
    sys.exit(-1)
  if not os.path.exists(label_file):
    print("label not exist: %s" % label_file)
    sys.exit(-1)

  list_mode = 1 if os.path.isfile(image_list) else 0
  path_test = image_list

  basename_to_fullpath = {}
  os.environ["CUDA_VISIBLE_DEVICES"] = str(notebook_util.pick_gpu_lowest_memory())
 
  # 构建labels
  labels = []
  with open(label_file, "r") as f:
    for line in f:
      labels.append(line.strip())

  # 加载model
  # 输出的softmax前加mean_max_pooling层
  # https://github.com/cypw/DPNs
  # https://github.com/titu1994/Keras-DualPathNetworks/blob/master/dual_path_network.py#L461
  if FLAGS.mean_max_pooling:
    print("Using Mean_Max Pooling before softmax")
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, add, Lambda
    from tensorflow.keras.models import Model
    base_model = InceptionV3(weights=None, include_top=False) # 目前仅支持Inception_V3
    x = base_model.output
    a = GlobalAveragePooling2D()(x)
    b = GlobalMaxPooling2D()(x)
    x = add([a, b])
    x = Lambda(lambda z: 0.5 * z)(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(labels), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(model_file, by_name=True)
    model.name = "inception_v3"
  else:
    from tensorflow.keras.models import load_model
    model = load_model(model_file)

  if image_size == 0:
    from retrain_full import get_image_size
    image_size = get_image_size(model) 
  from retrain_full import get_preprocess_input
  preprocess_input = get_preprocess_input(model)
  generator = SMSequence(directory=path_test,
      list_mode=list_mode,
      batch_size=batch_size,
      image_size=image_size,
      interpolation=interpolation,
      letter_box=letter_box,
      preprocessing_function=preprocess_input,
      shuffle=False)
  print('class indicses: %s' % generator.class_indices)
  count_test = len(generator.filenames)
  
  generator_idx_2_model_idx = [-1] * len(labels)
  for key, value in generator.class_indices.items():
    assert key in labels
    generator_idx_2_model_idx[value] = labels.index(key)

  starttime = datetime.datetime.now()  
  y, y_pred = calc_pred_and_labels_complete(generator, model, multiprocessing)
  endtime = datetime.datetime.now()
  seconds = (endtime - starttime).total_seconds()
  print("seconds: %d, speed: %.2f imgs/s" % (seconds, count_test/float(seconds)))

  # 将y中的label idx转换为labels中的label idx
  for i, label_idx in enumerate(y):
    y[i] = generator_idx_2_model_idx[label_idx]
    assert y[i] != -1
  # 将多分类按照映射表输出类别更少的多分类结果
  if classes_map:
    classes_map_dict = {}
    for row in open(classes_map, 'r'):
      assert len(row.strip().split()) == 2
      ori_label, new_label = row.strip().split()
      classes_map_dict[labels.index(ori_label)] = new_label
    labels = sorted(set(classes_map_dict.values()))

  # 回归预测结果
  y_pred_reg = []
  # 输出预测结果
  y_new = [] # 类别降维后的真实标签
  y_pred_new = [] #类别降维后的预测概率
  with open(output, "w") as f:
    f.write('%d\n' % len(labels))
    for i, item in enumerate(labels):
      f.write('%d %s\n' % (i, item)) 
    for i in range(count_test):
      if len(basename_to_fullpath) == 0:
        fullpath = generator.filenames[i]
      else:
        basename = os.path.basename(generator.filenames[i])
        assert basename_to_fullpath.has_key(basename)
        fullpath = basename_to_fullpath[basename]
      y_all = y_pred[i].tolist()
      max_idx = y_all.index(max(y_all)) # by argmax
      if optimizing_type == 'regression':
        max_idx = round(y_all[0]) # 将回归作为分类来评估
        y_pred_reg.append(y_all[0]) # 保存回归结果
        f.write('%s\t%d\t' % (fullpath, max_idx))
        for prob in y_all:
          f.write('%f\t' % prob)
      if optimizing_type == 'classify':
        if classes_map:
          max_idx = labels.index(classes_map_dict[max_idx])
          y_all_new = [0.] * len(labels)
          for idx in range(len(y_all)):
            y_all_new[labels.index(classes_map_dict[idx])] += y_all[idx]
          y_all = y_all_new
          y_pred_new.append(y_all)
          if not keep_pred:
            max_idx = y_all.index(max(y_all))
        assert len(y_all) == len(labels)
        f.write('%s\t%d\t' % (fullpath, max_idx))
        for prob in y_all:
          f.write('%f\t' % prob)
      if classes_map:
        gt = labels.index(classes_map_dict[y[i]])
        f.write('%d\n' % gt)
        y_new.append(gt)
      else:
        f.write('%d\n' % y[i])
  
  # 计算每一类的AP和AUC
  if optimizing_type == 'classify':
    if classes_map:
      y = y_new
      y_pred = np.array(y_pred_new)
    #ap_all, auc_all = calc_ap_and_auc(y, y_pred)
    #for i, label in enumerate(labels):
    #  print('label %s: AP=%.4f, AUC=%.4f' % (label, ap_all[i], auc_all[i]))
    compare_with_other(output, '', '', special, optimizing_type, find_best_threshold=find_best_threshold)
    return None#(ap_all, auc_all)
  elif optimizing_type == 'regression':
    mse_all = mean_squared_error(y, y_pred_reg)
    print('MSE=%.4f' % (mse_all))
    compare_with_other(output, '', '', special, optimizing_type, find_best_threshold=0)
    return mse_all
  return None

# keep the idxs in list correct_idx for lbl, pred, pred_lbl and fullpath
def filter_result(lbl, pred, pred_lbl, fullpath, correct_idx):
  correct_lbl = [lbl[x] for x in correct_idx]
  correct_pred = [pred[x] for x in correct_idx]
  correct_pred_lbl = [pred_lbl[x] for x in correct_idx]
  correct_fullpath = [fullpath[x] for x in correct_idx]
  return (correct_lbl, correct_pred, correct_pred_lbl, correct_fullpath)

# search diff
def search_diff(lbl, lbl2, pred_lbl, pred_lbl2, fullpath, fullpath2):
  diff_lbl = []
  diff_pred_lbl = []
  diff_pred_lbl2 = []
  img_dict = {}
  img_dict2 = {}
  for i in range(len(lbl)):
     basename = os.path.basename(fullpath[i])
     img_dict[basename] = {"lbl": lbl[i], "pred_lbl": pred_lbl[i]}
     basename2 = os.path.basename(fullpath2[i])
     img_dict2[basename2] = {"lbl": lbl2[i], "pred_lbl": pred_lbl2[i]}

  for img, v in img_dict.items():
    if v["pred_lbl"] != img_dict2[img]["pred_lbl"]:
      diff_lbl.append(v["lbl"])
      diff_pred_lbl.append(v["pred_lbl"])
      diff_pred_lbl2.append(img_dict2[img]["pred_lbl"])
  return diff_lbl, diff_pred_lbl, diff_pred_lbl2

# 寻找最优阈值
def preprocess_for_threshold(lbl, pred, labels, special=''):
  if len(labels) == 2:
    return find_best_threshold(lbl, [float(p[1]) for p in pred])
  if special == 'porn':
    for idx, lb in enumerate(labels):
      if lb == 'porn':
        OBJ_IDX = idx
    assert OBJ_IDX != -1
    return find_best_threshold([l if l != 2 else 0 for l in lbl], [float(p[OBJ_IDX]) for p in pred])
  elif special == 'violence':
    for idx, lb in enumerate(labels):
      if lb == 'zhengchang':
        NORMAL_IDX = idx
    assert NORMAL_IDX != -1
    stg_lbl = [0 if lb == NORMAL_IDX else 1 for lb in lbl]
    stg_pred = []
    for plist in pred:
      stg_pred.append(max([float(p) if idx != NORMAL_IDX else 0.0 for idx, p in enumerate(plist)]))
    return find_best_threshold(stg_lbl, stg_pred)
        
# 根据F1 score，返回最优阈值
def find_best_threshold(y, pred, th=0.0):
  f1_dict = {}
  with open("threshold_info.txt", "w") as f:
    f.write("threshold\tprecision\trecall\tf1 score\n")
    while th < 1.0:
      y_pred = [1 if p >= th else 0 for p in pred]
      result = precision_recall_fscore_support(y, y_pred)
      precision = result[0][1]
      recall = result[1][1]
      f1 = result[2][1]
      f1_dict[f1] = th
      f.write("{}\t{}\t{}\t{}\n".format(th, precision, recall, f1))
      th += 0.01
    best_f1 = max(f1_dict.keys())
    f.write("best F1_score {} with threshold {}\n".format(best_f1, f1_dict[best_f1]))
  return f1_dict[best_f1], best_f1

# 和第三方的结果进行简单对比（对应AUC曲线上的一个点），准确率、召回率和F1
def compare_with_other(shumei_result, other_result, file_filter='', special='', optimizing_type='classify', find_best_threshold=0):
  lbl, pred, pred_lbl, fullpath, labels = read_file(shumei_result, file_filter)
  labels_idx = [i for i in range(len(labels))]
  fullpath = [os.path.basename(x) for x in fullpath]
  set1 = set(fullpath)
  if other_result != '':
    lbl2, pred2, pred_lbl2, fullpath2, _ = read_file(other_result, file_filter)
    fullpath2 = [os.path.basename(x) for x in fullpath2]
    set2 = set(fullpath2)
    intersection = set1 & set2
    print('Files only available for Shumei:')
    print(set1 - set2)
    print('Files only available for Other:')
    print(set2 - set1)
    idx1 = {}
    idx2 = {}
    for i in range(len(fullpath)):
      if fullpath[i] in intersection and fullpath[i] not in idx1:
        idx1[fullpath[i]] = i
    for i in range(len(fullpath2)):
      if fullpath2[i] in intersection and fullpath2[i] not in idx2:
        idx2[fullpath2[i]] = i
    lbl, pred, pred_lbl, fullpath = filter_result(lbl, pred, pred_lbl, fullpath, idx1.values())
    lbl2, pred2, pred_lbl2, fullpath2 = filter_result(lbl2, pred2, pred_lbl2, fullpath2, idx2.values())
  
  print('precision recall fbeta_score')
  result1 = precision_recall_fscore_support(lbl, pred_lbl, labels=labels_idx)
  shumei_all_acc = accuracy_score(lbl, pred_lbl)
  print('labels:')
  print('\t'.join(labels))
  print('Shumei:')
  for item in result1:
    print('\t'.join(str(x) for x in item.tolist()))
  print('Shumei ACC:', shumei_all_acc)
  # 线上策略指标
  eval_on_tactics(shumei_result, file_filter, special)
  if optimizing_type == 'classify':
    th, best_f1 = preprocess_for_threshold(lbl, pred, labels, special='porn')
    print("best F1_score {} with threshold {}".format(best_f1, th))
  if other_result != '':
    result2 = precision_recall_fscore_support(lbl2, pred_lbl2, labels=labels_idx)
    other_all_acc = accuracy_score(lbl2, pred_lbl2)
    print('Other:', result2)
    print('Other ACC:', other_all_acc)
    diff_lbl, diff_pred_lbl, diff_pred_lbl2 = search_diff(lbl, lbl2, pred_lbl, pred_lbl2, fullpath, fullpath2)
    print("diff case num:", len(diff_lbl))
    print("Shumei ACC on diff case:", accuracy_score(diff_lbl, diff_pred_lbl))
    print("Other ACC on diff case:", accuracy_score(diff_lbl, diff_pred_lbl2))
     
'''
根据线上策略计算色情指标
4分类标签顺序:serious_porn(exposed)|normal|porn|sexy
3分类标签顺序:normal|porn|sexy
3分类线上策略:
  色情:porn > 0.8 or (normal < 0.2 and porn > sexy)
  性感:normal < 0.2 and sexy > porn
  正常:不满足上述两条中任一规则
4分类线上策略:
  重色:serious_porn > 0.8 否则将重色概率加入到porn，然后按照3分类策略进行判别.
'''
def eval_on_tactics(inference, file_filter='', special='porn'):
  if special != 'porn':
    return None
  lbl, pred, pred_lbl, fullpath, labels = read_file(inference, file_filter)
  labels_idx = [i for i in range(len(labels))]
  for i in range(len(lbl)):
    probs = [float(p) for p in pred[i]]
    if len(probs) == 4:
      if probs[0] > 0.8:
        pred_lbl[i] = 0
      else:
        probs[2] += probs[0]
        pred_lbl[i] = porn3_on_tactics(probs[1:]) + 1
    else:
      pred_lbl[i] = porn3_on_tactics(probs)
  result = precision_recall_fscore_support(lbl, pred_lbl, labels=labels_idx)
  shumei_all_acc = accuracy_score(lbl, pred_lbl)
  print("线上策略评估结果")
  print('labels:', labels)
  print('Shumei:', result)
  print('Shumei ACC:', shumei_all_acc)

# 线上色情三分类策略
# 3分类标签顺序:normal|porn|sexy
def porn3_on_tactics(probs):
  assert len(probs) == 3
  if probs[1] > 0.8 or (probs[0] < 0.2 and probs[1] > probs[2]):
    return 1
  elif probs[0] < 0.2 and probs[2] > probs[1]:
    return 2
  else:
    return 0
 
# plot and save roc-curve and pr-curve, print auc 
def plot_curves(y_true, y_scores, y_true2 = None, y_scores2 = None):
  comp = False
  control_point = 0.99
  fpr, tpr, _ = roc_curve(y_true, y_scores)
  precision, recall, _ = precision_recall_curve(y_true, y_scores)
  roc_auc = auc(fpr, tpr)
  roc_auc2 = 0
  fpr_at_cp = interpolate.interp1d(tpr, fpr)(control_point)
  #fpr_at_cp = calc_fpr_at(tpr, fpr, control_point)
  
  # estimate best limit for fpr axis
  max_fpr1 = 1.0
  for i, item in enumerate(fpr):
    if tpr[i] == 1.0:
      max_fpr1 = item
      break
  
  # start plot
  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange',
           lw=lw, label='shumei (auc = %0.2f%%, fpr@0.99 = %0.1f%%)' %
           (roc_auc*100, fpr_at_cp*100))
 
  # add the other data
  if y_true2 != None and y_scores2 != None:
    fpr2, tpr2, _ = roc_curve(y_true2, y_scores2)
    precision2, recall2, _ = precision_recall_curve(y_true2, y_scores2)
    roc_auc2 = auc(fpr2, tpr2)
    #fpr_at_cp2 = calc_fpr_at(tpr2, fpr2, control_point)
    fpr_at_cp2 = interpolate.interp1d(tpr2, fpr2)(control_point)
    comp = True
 
    # plot the other curve
    plt.plot(fpr2, tpr2, color='red',
           lw=lw, label='baidu (auc = %0.2f%%, fpr@0.99 = %0.1f%%)' %
           (roc_auc2*100, fpr_at_cp2*100))
    max_fpr2 = 1.0
    for i, item in enumerate(fpr2):
      if tpr2[i] == 1.0:
        max_fpr2 = item
        break
    max_fpr1 = max(max_fpr1, max_fpr2)

  #默认x轴只显示到y=1.0，也可以自己设置
  max_fpr1 = 0.5
  plt.xlim([0.0, max_fpr1])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate (Recall)')
  plt.title('ROC for Porn')
  plt.legend(loc="lower right")
  plt.savefig("roc.jpg")
  plt.close()

  # pr-curve
  plt.figure()
  plt.plot(precision, recall, color='darkorange', lw=lw, label='shumei')
  if comp:
    plt.plot(precision2, recall2, color='red', lw=lw, label='baidu')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Precision')
  plt.ylabel('Recall')
  plt.title('Precision Recall Curve for Porn')
  plt.legend(loc="lower right")
  plt.savefig("pr-curve.jpg")
  plt.close()
  
  print("auc: %s" % roc_auc)
  if comp:
    print("auc2: %s" % roc_auc2)
  return (roc_auc, roc_auc2)

# 主函数
if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--model', type=str, default='',
    help='The hdf5 model used to get the inference result. If a directory is given, all models in the dir will be evaluated.'
  )
  parser.add_argument(
    '--labels', type=str, default='',
    help='The labels file for the model, can auto inferenced with the same dirname of the model with filename labels.txt.'
  )
  parser.add_argument(
    '--input', type=str, default='input.list',
    help='The input file list, have the line format of: full_path true_label,'
    'a folder of the same filename(e.g. input) will be created in current dir for inference;'
    'or a dir with labeled data just like prepare_data.py generated, e.g. tmp/test.'
  )
  parser.add_argument(
    '--output', type=str, default='inference_output.txt',
    help='The inference output file, ref Porn-Dev doc for detailed format of this file.'
    'The output will not generate if set to an empty string.'
  )
  parser.add_argument(
    '--eval_other', type=str, default='',
    help='The inference_output.txt of other API for evaluation.'
  )
  parser.add_argument(
    '--eval_shumei', type=str, default='',
    help='The inference_output.txt of Shumei API for evaluation.'
  )
  parser.add_argument(
    '--eval_filter', type=str, default='',
    help='A filter of fullpath to evaluate a sub set from inference_output.'
  )
  parser.add_argument(
    '--multiprocessing', type=int, default=0,
    help='Whether turn on the multiprocessing optimization.'
  )
  parser.add_argument(
    '--interpolation', type=str, default='bilinear',
    help='Interpolation method used to resample the image if the target size is different from that of the loaded image, e.g:bilinear, bicubic, lanczos, antialias. if input "random", it will pick interpolation method randomly'
  )
  parser.add_argument(
    '--letter_box', type=int, default=0,
    help='Scale with keeping aspect ratio when letter box is opened, padding black on margin.'
  )
  parser.add_argument(
    '--batch_size', type=int, default=0,
    help='The batch_size to use, 256 will be used if a default of 0 is given.'
  )
  parser.add_argument(
    '--special', type=str, default='',
    help='Special process for different tasks, e.g. porn/violence/fit.'
  )
  parser.add_argument(
    '--optimizing_type', type=str, default='classify',
    help='The type of optimizing, e.g. classify/regression.'
  )
  parser.add_argument(
    '--find_best_threshold', type=str, default=0,
    help='Whether to find best thershold in porn or violence, e.g. 0/1.'
  ) 
  parser.add_argument(
    '--image_size', type=int, default=0,
    help='define the image_size of the model'
  )
  parser.add_argument(
    '--classes_map', type=str, default="",
    help='Input a list file to realize classes num reduction, e.g. "porn_6classes_to_4classes.list"'
  )
  parser.add_argument(
    '--keep_pred', type=int, default=1,
    help='Whether to keep old pred label, else save new argmax label on less classes.'
  )
  parser.add_argument(
    '--mean_max_pooling', type=int, default=0,
    help='Whether to insert Mean-Max Pooling layer which before the final softmax layer.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  print(FLAGS)
  if len(unparsed) > 0:
    print("there are unknow args %s " % ','.join(unparsed))
    sys.exit(-1)
  
  model = FLAGS.model
  labels = FLAGS.labels
  batch_size = 256
  special = FLAGS.special
  optimizing_type = FLAGS.optimizing_type
  image_size = FLAGS.image_size
  interpolation = FLAGS.interpolation
  letter_box = FLAGS.letter_box
  classes_map = FLAGS.classes_map
  keep_pred = FLAGS.keep_pred
  if FLAGS.batch_size > 0:
    batch_size = FLAGS.batch_size
  multiprocessing = FLAGS.multiprocessing
  if model != "":
    input_list = FLAGS.input
    output = FLAGS.output
    if model.endswith('.hdf5'):
      # 对指定模型进行评估
      if labels == "":
        labels = os.path.join(os.path.dirname(model), 'labels.txt')
      eval_model(model, labels, input_list, output, 
          multiprocessing, image_size, interpolation=interpolation, letter_box=letter_box, batch_size=batch_size, special=special, classes_map=classes_map, keep_pred=keep_pred, optimizing_type=optimizing_type)
    else:
      print("model must endswith hdf5")
      sys.exit(-1)
      '''
      # 自动推断labels.txt
      if labels == "":
        labels = os.path.join(model, 'labels.txt')
      # 对该目录下所有模型进行评估
      files = os.listdir(model)
      files = sorted(files)
      model_to_eval = []
      for f in files: 
        if f.endswith('.hdf5'):
          print('##### eval auc for model: %s' % f)
          fullpath = os.path.join(model, f)
          if optimizing_type == "classify":
            ap, auc = eval_model(fullpath, labels, input_list, output,
              multiprocessing, image_size, interpolation=interpolation, letter_box=letter_box, batch_size=batch_size, special=special, classes_map=classes_map, keep_pred=keep_pred, optimizing_type=optimizing_type)
            model_to_eval.append((f, ap, auc))
          elif optimizing_type == "regression":
            mse = eval_model(fullpath, labels, input_list, output,
              multiprocessing, image_size, interpolation=interpolation, letter_box=letter_box, batch_size=batch_size, special=special, classes_map=classes_map, keep_pred=keep_pred, optimizing_type=optimizing_type)
            model_to_eval.append((f, mse))
      if optimizing_type == "classify":
        for t in model_to_eval:
          print('model: %s, ap: %s, auc: %s' % (t[0], t[1], t[2]))
      elif optimizing_type == "regression":
        for t in model_to_eval:
          print('model: %s, mse: %s' % (t[0], t[1]))
      '''
  else:
    other_result = FLAGS.eval_other
    shumei_result = FLAGS.eval_shumei
    eval_filter = FLAGS.eval_filter
    compare_with_other(shumei_result, other_result, special=special, optimizing_type=optimizing_type, find_best_threshold=find_best_threshold)
