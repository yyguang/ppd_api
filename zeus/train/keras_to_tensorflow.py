#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <cao-teng-teng.cao@foxmail.com> 2017-10-18 16:27:31

from __future__ import print_function
import sys
import shutil
import notebook_util
import os


# 用法示例：/data/project/anaconda2/bin/python keras_to_tensorflow.py mt-180122finetune_201802061842/model.07-0.2426.hdf5
# 用法示例：/data/project/anaconda2/bin/python keras_to_tensorflow.py mt-180122finetune_201802061842/model.07-0.2426.hdf5 299
# 输出的log中包括输入节点名称（input tensor name）和输出节点名称（output tensor name）
if __name__ == '__main__':
  # Set parameters
  os.environ["CUDA_VISIBLE_DEVICES"] = "" #str(notebook_util.pick_gpu_lowest_memory()) 
  input_fld = sys.argv[1]   # 'input_fld_path'
  input_fld, weight_file = os.path.split(sys.argv[1]) 
  num_output = 1
  write_graph_def_ascii_flag = True
  output_graph_name = 'output_graph.pb'

  # initialize
  from tensorflow.keras.models import load_model
  import tensorflow as tf
  import os.path as osp
  from tensorflow.keras import backend as K

  output_fld = osp.join(input_fld, 'tensorflow_model')
  if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
  weight_file_path = osp.join(input_fld, weight_file)

  # Load keras model and rename output
  K.set_learning_phase(0)
  net_model = load_model(weight_file_path)

  try:
    size = int(sys.argv[2])
    assert size != 0
  except:
    if net_model.name == "inception_v3":
      size = 299
    elif net_model.name == "mobilenetv2_1.40_224":
      size = 224
    else:
      size = 299
      print("Warning: model name: %s not currently supported, please supplement the size of this model" % net_model.name)
      print("using default size: %s" % size)

  print('input tensor name:', net_model.input.name[:-2])
  print('output tensor name:', net_model.output.name[:-2])
  pred_node_names = [net_model.output.name[:-2]]

  sess = K.get_session()

  # convert variables to constants and save
  from tensorflow.python.framework import graph_util
  from tensorflow.python.framework import graph_io
  constant_graph = graph_util.convert_variables_to_constants(sess,
      sess.graph.as_graph_def(), pred_node_names)
  # [optional] write graph definition in ascii
  if write_graph_def_ascii_flag:
    f = 'only_the_graph_def.pb.ascii'
    graph_io.write_graph(constant_graph, output_fld, f, as_text=True)
    print('saved the graph definition in ascii format at: ', osp.join(output_fld, f))
  graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
  print('saved the constant graph (ready for inference) at: ',
      osp.join(output_fld, output_graph_name))
  
  # save OpenVINO IR format output
  pb_path = os.path.abspath(os.path.join(output_fld, output_graph_name))
  pb_root = os.path.dirname(pb_path)
  os.chdir('/data/project/intel/computer_vision_sdk/deployment_tools/model_optimizer')
  os.system('/data/project/anaconda3-v3.0/bin/python mo_tf.py --input_model %s --input_shape [1,%s,%s,3] --scale 127.5 --mean_values [127.5,127.5,127.5] --reverse_input_channels --output_dir %s' % (pb_path, size, size, pb_root))
  print('saved the OpenVINO IR format at: ', pb_root)
