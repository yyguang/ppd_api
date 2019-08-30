# -*- coding: utf-8 -*-

"""
@author:yunkchen
@file:upload_label_task.py
@time:2017/11/23 14:32
"""
import os
import sys
import pickle
import imghdr


def check_file(img_dir):
  assert(os.path.exists(img_dir))
  """检查文件夹是否包含隐藏文件"""
  for f in os.listdir(img_dir):
    if f.startswith('.'):
      print("{} contains hidden file {}".format(img_dir, f))
      return False
  return True

def imgInfo(img):
  assert(os.path.exists(img))
  return imghdr.what(img)

def upload_imgs(img_dir="test_pics"):
  try:
    os.system("scp -r {} imgupload@admin1.bj.sm:/mnt/imgupload/".format(img_dir))
  except Exception as e:
    print(e)


def creat_task_list(img_dir="test_pics", dir_label_rule=True, is_hash_sort=False):
  if not is_hash_sort:
    imgs = os.popen('find {} -type f -name "*.jpg"| sort -r'.format(img_dir)).read().strip().split("\n")
    
    imgs = ['/'.join(img.split('/')[-3:]) for img in imgs]
  else:
    img_dir_path = os.path.join(os.getcwd(), img_dir)
    sim_families_list_path = os.path.join(os.getcwd(), 'sim_families_list.var')
    os.popen(
      'python ../hash_clean/hash_clean.py \
      --mode label_task \
      --hash_type {} \
      --dataset_path {} \
      --hamming_dist_thr {} \
      --sim_families_list_path {} \
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
      for img_attr in sim_fam:
        imgs.append('/'.join(img_attr['path'].split('/')[-3:]))
  urls = []
  if dir_label_rule:
    for img in imgs:
      try:
        print(img)
        if imgInfo(img) != 'gif':
          urls.append("https://data.fengkongcloud.com/image/{},{}".format(img, img.split("/")[-2]))
        else:
          print("ignore gif image :{}".format(img))
      except Exception as e:
        print("{}:{}".format(e, img))
  else:
    for img in imgs:
      if imgInfo(img) != 'gif':
        urls.append("https://data.fengkongcloud.com/image/{}".format(img))
      else:
        print("ignore gif image :{}".format(img)) 
  with open("/tmp/{}_list".format(os.path.basename(img_dir.strip().strip('/'))), "w") as f:
    f.write("\n".join(urls)+"\n")


def task_submit(img_dir="test_pics", objective="category", per_task_count=300):
  try:
    os.system('curl --data-binary "@/tmp/{}" -XPOST '
              '"https://webapi-bj-bd.fengkongcloud.com/DataManager/annotation/import?'
	      'organization=9h4YLrU1SDTN7c2srruX&accessKey=qaVCztDBDAGdhaBt2WUl&serviceId=POST_IMG&perTaskCount={}&taskType='
              '{}"'.format(os.path.basename(img_dir.strip().strip('/'))+"_list", per_task_count,  objective))
    print()
  except Exception as e:
    print(e)


if __name__ == '__main__':
  if len(sys.argv) != 5:
    print(sys.argv[0] + ' img_dir task_type per_task_count ishash_sort')
    print('''task_type:
        category\n\
        category-fit\n\
        category-violence\n\
        category-general-ad\n\
        category-grindr\tvline/man_bareness/convex/other\n\
        category-politics\n\
        rectangle\n\
        rectangle-with-text\n\
        rectangle-with-label\n\
        violent_terrorist\n\
        quadrilateral-with-text\n\
        rectangle-quadrilateral-with-text''')
    print('per_task_count: 每个任务编号图片的个数')
    print('ishash_sort: 是否进行 hash 相似排序,1或者0')
    sys.exit(-1)
  img_dir = sys.argv[1]
  objective = sys.argv[2]
  per_task_count = int(sys.argv[3])
  is_hash_sort = int(sys.argv[4])
  os.system('chmod -R 777 ' + img_dir)
  if check_file(img_dir):
    upload_imgs(img_dir)
    creat_task_list(img_dir, objective.startswith("category") or 
        objective == 'violent_violence', is_hash_sort=is_hash_sort)
    task_submit(img_dir, objective, per_task_count)
