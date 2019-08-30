#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <teng.cao@foxmail.com> 2018-11-03 10:34:11
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint

class MultiModelCheckpoint(ModelCheckpoint):
  def set_model(self, model):
    if isinstance(model.layers[-2], keras.models.Model):
      self.model = model.layers[-2]
      print("Using last but one layer as model in callbacks.")
    else:
      self.model = model
