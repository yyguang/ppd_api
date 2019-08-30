#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <teng.cao@foxmail.com> 2018-11-01 19:03:51
import tensorflow.keras as keras
import datetime
import os
from evaluation import calc_pred_and_labels_complete
from evaluation import calc_ap_and_auc, get_pred_label, calc_precision_recall_f1 
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K
from tensorflow.data import Iterator

# get a dict of class indices map
def get_class_indices_dict(classes_map, class_indices):
  assert os.path.exists(classes_map)
  classes_map_dict = {}
  class_indices_dict = {}
  class_indices_new = {}
  for row in open(classes_map, 'r'):
    ori_label, new_label = row.strip().split()
    classes_map_dict[ori_label] = new_label
  idx = 0
  for lbl in set(sorted(classes_map_dict.values())):
    class_indices_new[lbl] = idx
    idx += 1
  print('new class indices: ', class_indices_new)
  for ori_lbl, new_lbl in classes_map_dict.items():
    class_indices_dict[class_indices[ori_lbl]] = class_indices_new[new_lbl]
  return class_indices_dict

# callback to calculate auc
class RocCallback(keras.callbacks.Callback):
  def __init__(self, train_generator, test_generator, validation_generator,
      train_validation_files, classes_map, multiprocessing, workers, validation_auc = 0, test_auc = 1, history_metrics = [], early_stopping_by_val_acc = 0):
    self.train_generator = train_generator
    self.test_generator = test_generator
    self.validation_generator = validation_generator
    self.train_validation_files = train_validation_files
    self.validation_auc = validation_auc
    self.test_auc = test_auc
    self.metrics = []
    self.history_metrics = history_metrics
    self.best_model_idx = -1
    #self.classes_map = classes_map
    self.class_indices_map = {}
    self.early_stopping_by_val_acc = early_stopping_by_val_acc
    if classes_map:
      self.class_indices_map = get_class_indices_dict(classes_map, self.test_generator.class_indices) 
    self.multiprocessing = multiprocessing
    self.workers = workers

  def on_train_begin(self, logs={}):
    return
 
  def on_train_end(self, logs={}):
    return
 
  def on_epoch_begin(self, epoch, logs={}):
    self.starttime = datetime.datetime.now()
    if isinstance(self.train_generator, Iterator):
      sess = K.get_session()
      sess.run(self.train_generator.initializer)
      sess.run(self.test_generator.initializer)
      sess.run(self.validation_generator.initializer)
    return
 
  def on_epoch_end(self, epoch, logs={}):
    endtime = datetime.datetime.now()
    seconds = (endtime - self.starttime).seconds
    metric = {}
    metric['seconds'] = seconds
     
    from sklearn.metrics import roc_auc_score
    starttime = datetime.datetime.now()
    if self.test_auc:
      y, y_pred = calc_pred_and_labels_complete(self.test_generator,
          self.model, self.multiprocessing, self.workers)
      #ap, auc = calc_ap_and_auc(y, y_pred)
      #metric['test_auc'] = ap + auc
      P, R, F1 = calc_precision_recall_f1(y, y_pred)
      metric['test_auc'] = P + R + F1
      P, R, F1 = calc_precision_recall_f1(y, y_pred, self.class_indices_map)
      metric['new_class_auc'] = P + R + F1
      pred_lbl = get_pred_label(y_pred)
      metric['test_acc'] = accuracy_score(y, pred_lbl)
    else:
      metric['test_auc'] = [-1]
      metric['test_acc'] = -1
    endtime = datetime.datetime.now()
    seconds = (endtime - starttime).seconds
    metric['test_seconds'] = seconds

    if self.validation_auc:
      y, y_pred = calc_pred_and_labels_complete(self.validation_generator,
          self.model, self.multiprocessing, self.workers)
      P, R, F1 = calc_precision_recall_f1(y, y_pred)
      metric['val_auc'] = P + R + F1
      P, R, F1 = calc_precision_recall_f1(y, y_pred, self.lass_indices_map)
      metric['new_class_auc'] = P + R + F1
      #ap, auc = calc_ap_and_auc(y, y_pred)
      #metric['val_auc'] = ap + auc
    else:
      metric['val_auc'] = [-1]
    
    metric['loss'] = logs.get('loss')
    metric['val_loss'] = logs.get('val_loss')
    metric['acc'] = logs.get('acc')
    metric['val_acc'] = logs.get('val_acc')
    self.metrics.append(metric)

    for i, history in enumerate(self.history_metrics):
      print('history for stage %d' % (i+1))
      self.print_infor(history)
    print('metrics for current stage')
    self.print_infor(self.metrics)
    if self.early_stopping_by_val_acc and metric['val_acc'] >= self.early_stopping_by_val_acc:
      print('Epoch %s: early stopping by val_acc %s, current val acc: %s' % (epoch+1, self.early_stopping_by_val_acc, metric['val_acc']))
      self.model.stop_training = True
    return
 
  def on_batch_begin(self, batch, logs={}):
    return
 
  def on_batch_end(self, batch, logs={}):
    return

  def print_infor(self, metrics):
    str_infor = []
    if self.class_indices_map:
      print('epoch loss val_loss acc val_acc test_acc time speed val_metric test_metric new_class_metric')
    else:
      print('epoch loss val_loss acc val_acc test_acc time speed val_metric test_metric')
    for i, metric in enumerate(metrics):
      speed = self.train_validation_files / float(metric['seconds'])
      val_aucs_str = ['%.4f' % x for x in metric['val_auc']]
      aucs_str = ['%.4f' % x for x in metric['test_auc']]
      concat_aucs_str = "|".join(aucs_str)
      tmp_str = '%.4f, %.4f, %.4f, %.4f, %.4f, %d, %.1f, %s, %s' % (metric['loss'],
        metric['val_loss'], metric['acc'], metric['val_acc'], metric['test_acc'],
        metric['seconds'], speed, "|".join(val_aucs_str), concat_aucs_str)
      if self.class_indices_map:
        tmp_str += ", {}".format("|".join(['%.4f' % x for x in metric['new_class_auc']]))
      str_infor.append(tmp_str)
      print('%02d: %s' % (i + 1, str_infor[i]))
    
    # print epoch idx start from 1
    val_accs = [metric['val_acc'] for metric in metrics]
    max_acc_idx = val_accs.index(max(val_accs))
    print('best val_acc  @%d, %s' % (max_acc_idx+1, str_infor[max_acc_idx]))
    val_losses = [metric['val_loss'] for metric in metrics]
    min_val_loss_idx = val_losses.index(min(val_losses))
    print('best val_loss @%d, %s' % (min_val_loss_idx+1, str_infor[min_val_loss_idx]))
    self.best_model_idx = min_val_loss_idx
    if self.validation_auc:
      val_aucs = [metric['val_auc'] for metric in metrics]
      max_val_auc_idx = val_aucs.index(max(val_aucs))
      print('best val_auc @%d, %s' % (max_val_auc_idx+1, str_infor[max_val_auc_idx]))
