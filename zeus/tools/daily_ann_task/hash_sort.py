# -*- coding: UTF-8 -*-

import os
import sys
import pickle

task_label = {"ad" : "daily_ad", \
              "politics" : "daily_politics", \
              "smoke" : "daily_smoke", \
              "violence" : "daily_violence", \
              "porn" : "daily_porn"}


def creat_task_list(task_type, img_dir="test_pics", dir_label_rule=False, is_hash_sort=False):
  if not is_hash_sort:
    imgs = os.popen('find {} -type f -name "*.jpg"| sort -r'.format(img_dir)).read().strip().split("\n")
    imgs = ['/'.join(img.split('/')[-3:]) for img in imgs]
  else:
    img_dir_path = os.path.join(os.getcwd(), img_dir)
    sim_families_list_path = os.path.join(os.getcwd(), 'sim_families_list.var')
    os.popen(
      'python ../../hash_clean/hash_clean.py \
      --mode label_task \
      --hash_type {} \
      --dataset_path {} \
      --hamming_dist_thr {} \
      --sim_families_list_path {}\
      '.format('phash', img_dir_path, '10', sim_families_list_path)).read()
    with open(sim_families_list_path, 'rb') as var_file:
      sim_families_list = pickle.load(var_file)
    os.popen('rm {}'.format(sim_families_list_path))
    if len(sim_families_list) == 0:
      print('''
      Error: please check img_dir. it must be
          <dataset>
          └── <label>
               └── <img>
      ''')
      exit(-1)
    imgs = []
    for sim_fam in sim_families_list:
      for dirty_img_attr in sim_fam:
        imgs.append('/'.join(dirty_img_attr['path'].split('/')[-3:]))
  urls = []
  if False:
    for img in imgs:
      try:
        urls.append("https://data.fengkongcloud.com/image/"+str(task_label[task_type])+"/{},{}".format(img, img))
      except Exception as e:
        print("{}:{}".format(e, img))
  else:
    for img in imgs:
      urls.append("https://data.fengkongcloud.com/image/"+str(task_label[task_type])+"/{}".format(img))

  for url in urls:
    print url

img_dir = sys.argv[1]
task_type = sys.argv[2]

creat_task_list(task_type, img_dir, False, is_hash_sort=True)

