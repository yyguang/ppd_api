#encoding:utf-8
# Author fengjianming <869800526@qq.com> 2018-10-12 17:00:00
import os
import hashlib
import sys
import argparse
import random
from PIL import Image, ImageFilter
import numpy as np
import time

paste_location = ['global_region','left_top','right_top','left_bottom','right_bottom']
relative_area_thresh = 32.0*32.0/(416.0*416.0)
distance_thresh = (32**2+32**2+32**2)**0.5
similary_thresh = 0.5
success = 0
Buffer = ''

def get_md5(path):
  with open(path,'rb') as f:
    md5 = hashlib.md5()
    md5.update(f.read())
  return md5.hexdigest()

def reshape_relative_area(background_image, paste_image_process, paste_gt_process):
  global relative_area_thresh
  w_bg, h_bg = background_image.size
  w_fg, h_fg = paste_image_process.size
  background_area = float(w_bg*h_bg)
  paste_area = float(w_fg*h_fg)
  rate = paste_area/background_area
  if rate < relative_area_thresh:
    reshape_area = relative_area_thresh*background_area
    zoom = (reshape_area/paste_area)**0.5
    size = (int(w_fg*zoom),int(h_fg*zoom))
    paste_image_process = paste_image_process.resize(size, Image.ANTIALIAS)
    gt = []
    for box in paste_gt_process:
      gt.append([box[0], int(box[1]*zoom), int(box[2]*zoom), int(box[3]*zoom), int(box[4]*zoom)])
    return paste_image_process, gt
  else:
    return paste_image_process, paste_gt_process

#计算两个box的重叠面积，box的定义形式为（左上角x坐标，左上角y坐标，右下角x坐标，右下角y坐标），避免粘贴两个logo的时候重叠导致看不清
def insert_area(box1,box2):
  x = [box1[0],box1[2],box2[0],box2[2]]
  y = [box1[1],box1[3],box2[1],box2[3]]
  #计算x方向上边的重叠长度
  if max([x[3]-x[0],x[1]-x[2]]) >= (x[1]-x[0]+x[3]-x[2]):
    #没有重叠
    coincide_x = 0
  elif x[0]<=x[2]<=x[3]<=x[1] or x[2]<=x[0]<=x[1]<=x[3]:
    #其中一个box的一条边包含了另一个box的一条边
    coincide_x = min([x[1]-x[0],x[3]-x[2]])
  else:
    #其中一个box的一条边和另一个box的一条边部分重叠
    coincide_x = (x[1]-x[0]+x[3]-x[2])-max([x[3]-x[0],x[1]-x[2]])
  #计算y方向上的重叠长度
  if max([y[3]-y[0],y[1]-y[2]]) >= (y[1]-y[0]+y[3]-y[2]):
    coincide_y = 0
  elif y[0]<=y[2]<=y[3]<=y[1] or y[2]<=y[0]<=y[1]<=y[3]:
    coincide_y = min([y[1]-y[0],y[3]-y[2]])
  else:
    coincide_y = (y[1]-y[0]+y[3]-y[2])-max([y[3]-y[0],y[1]-y[2]])
  return coincide_x*coincide_y

def similarity(img_bg, img_fg):
  assert img_bg.size == img_fg.size
  global distance_thresh
  global similary_thresh
  try:
    bg_datas = img_bg.getdata()
    fg_datas = img_fg.getdata()
  except:
    return -1
  similar_pixel = 0
  opaque_pixel = 0
  for datas in zip(bg_datas, fg_datas):
    bg_data = datas[0]
    fg_data = datas[1]
    if fg_data[3] != 0:
      distance = ((bg_data[0]-fg_data[0])**2 + (bg_data[1]-fg_data[1])**2 + (bg_data[2]-fg_data[2])**2)**0.5
      if distance < distance_thresh:
        similar_pixel = similar_pixel + 1
      opaque_pixel = opaque_pixel + 1
  similary_rate = float(similar_pixel)/float(opaque_pixel)
  if similary_rate > similary_thresh:
    return 1
  else:
    return 0

def denoising(paste_image,paste_gt):
  paste_image = paste_image.filter(ImageFilter.MedianFilter(3))
  paste_image = paste_image.filter(ImageFilter.DETAIL)
  return paste_image,paste_gt

def Gaussian_Blur(paste_image,paste_gt,radius):
  paste_image = paste_image.filter(ImageFilter.GaussianBlur(radius=radius))
  return paste_image,paste_gt

def transparency(paste_image,paste_gt,transparency_rate):
  datas = paste_image.getdata()
  newdatas = []
  for item in datas:
    tmp = (item[0], item[1], item[2], int(item[3]*transparency_rate))
    newdatas.append(tmp)
  paste_image.putdata(newdatas)
  return paste_image,paste_gt

def aspect_ratio(paste_image,paste_gt,aspect_ratio_rate):
  paste_image_size = paste_image.size
  size = (int(paste_image_size[0]*aspect_ratio_rate),int(paste_image_size[1]*aspect_ratio_rate))
  paste_image = paste_image.resize(size,Image.ANTIALIAS)
  gt = []
  for box in paste_gt:
    gt.append([box[0], int(box[1]*aspect_ratio_rate), int(box[2]*aspect_ratio_rate), int(box[3]*aspect_ratio_rate), int(box[4]*aspect_ratio_rate)])
  return paste_image,gt

def image_processing(paste_image, paste_gt):
  assert paste_image.mode == 'RGBA'
  global is_GaussianBlur_probability, radius_value, radius_probability, transparency_rate_value, transparency_rate_probability, aspect_ratio_rate_value, aspect_ratio_rate_probability
  is_GaussianBlur = np.random.choice([1, 0], p=is_GaussianBlur_probability)
  if is_GaussianBlur:
    radius = np.random.choice(radius_value, p=radius_probability)
    paste_image, paste_gt = Gaussian_Blur(paste_image, paste_gt, radius)
  transparency_rate = np.random.choice(transparency_rate_value, p=transparency_rate_probability) 
  paste_image, paste_gt = transparency(paste_image,paste_gt,transparency_rate)
  aspect_ratio_rate = np.random.choice(aspect_ratio_rate_value, p=aspect_ratio_rate_probability)
  paste_image, paste_gt = aspect_ratio(paste_image,paste_gt,aspect_ratio_rate)
  return paste_image,paste_gt

def pasting(background_image, paste_image_process, paste_gt_process, total_used):
  global paste_location, location_probability
  gt = []
  w_bg, h_bg = background_image.size
  w_fg, h_fg = paste_image_process.size
  feasible_w = w_bg - w_fg
  feasible_h = h_bg - h_fg
  continue_count = 0
  while (continue_count < 100):
    continue_count = continue_count + 1
    flag = 0
    location = np.random.choice(paste_location, p=location_probability)
    if location == 'global_region':
      x = random.randint(0, feasible_w)
      y = random.randint(0, feasible_h) 
    elif location == 'left_top':
      x = 0
      y = 0
    elif location[0] == 'right_top':
      x = feasible_w
      y = 0
    elif location[0] == 'left_bottom':
      x = 0
      y = feasible_h
    else:
      x = feasible_w
      y = feasible_h
    intend = (x, y, x + w_fg, y + h_fg)
    for used in total_used:
      if insert_area(used, intend) != 0:
        flag = -1
        break
    if flag == 0:
      break
  if flag == 0:
    try:
      img_bg = background_image.crop(intend)
    except:
      return -1, -1, -1
    is_similar = similarity(img_bg, paste_image_process)
    if is_similar == -1 or is_similar == 1:
      return -1, -1, -1
    try:
      background_image.paste(paste_image_process, (x, y), paste_image_process)
    except:
      return -1, -1, -1
    for box in paste_gt_process:
      gt.append([box[0], box[1]+x, box[2]+y, box[3]+x, box[4]+y])
    return [background_image, gt, intend]
  elif flag == -1:
    return -1, -1, -1

def paste_precessor(background_image_list,paste_image_gt_list):
  global paste_image_count_value, paste_image_count_probability, repeat_value, repeat_probability, bg_size_limit
  background_image_file = random.sample(background_image_list, 1)[0]
  try:
    background_image = Image.open(background_image_file)
    w_bg, h_bg = background_image.size
    if w_bg < bg_size_limit[0] or h_bg < bg_size_limit[1]:
      return -1
  except:
    return -1
  if background_image.mode != 'RGB':
    try:
      background_image = background_image.convert('RGB')
    except:
      return -1
  paste_image_count = np.random.choice(paste_image_count_value, p=paste_image_count_probability)
  if paste_image_count == 0:
    final_image = background_image
    final_boxes = [[0,0,0,0,0]]
    return [final_image, final_boxes]
  paste_images_gts = random.sample(paste_image_gt_list,paste_image_count)
  paste_gts = [eval(line.split()[1]) for line in paste_images_gts]
  paste_images = [Image.open(line.split()[0]) for line in paste_images_gts]
  final_boxes = []
  total_used = []
  for i in range(paste_image_count):
    paste_image = paste_images[i]
    paste_gt = paste_gts[i]
    repeat = np.random.choice(repeat_value, p=repeat_probability)
    for i in range(repeat):
      paste_image_process, paste_gt_process = image_processing(paste_image,paste_gt)
      paste_image_process, paste_gt_process = reshape_relative_area(background_image, paste_image_process, paste_gt_process)
      w_bg, h_bg = background_image.size
      w_fg, h_fg = paste_image_process.size
      try:
        assert w_bg > w_fg and h_bg > h_fg
      except:
        return -1
      background_image, gt, used = pasting(background_image, paste_image_process, paste_gt_process, total_used)
      if background_image == -1:
        return -1
      final_boxes.extend(gt)
      total_used.append(used)
  final_image = background_image
  return [final_image,final_boxes]

def save_data(final_data,save_dir):
  image_save_dir = os.path.join(save_dir,'tmp.jpg')
  final_image = final_data[0]
  assert not os.path.exists(image_save_dir)
  final_image.save(image_save_dir,'JPEG', quality = 100)
  md5 = get_md5(image_save_dir)
  md5_file = os.path.join(save_dir,md5+'.jpg')
  try:
    assert not os.path.exists(md5_file)
  except:
    os.remove(image_save_dir)
    return -1
  os.rename(image_save_dir, md5_file)
  final_boxes = str(final_data[1]).replace(' ','')
  global Buffer
  Buffer = Buffer + md5_file + '\t '+ final_boxes + '\n'
  return 0

def main_processor(background_image_list,paste_image_gt_list,save_dir):
  final_data = paste_precessor(background_image_list, paste_image_gt_list)
  if final_data == -1:
    return
  flag = save_data(final_data, save_dir)
  if flag == 0:
    global success
    success = success + 1
    print('Done Successfully: %s' % success)

if __name__ == '__main__':
  argparser = argparse.ArgumentParser(description='Paste a logo image on a background image')
  argparser.add_argument('--background_image_dir', type = str, help = 'the address of direcotry that save background image' )
  argparser.add_argument('--paste_image_gt_file', type = str, help = 'the train_boxes.list that include paste_image path and groundtruth')
  argparser.add_argument('--save_dir', type = str, help = 'the directory for saving generated images')
  argparser.add_argument('--total', type = int, default=2000, help = 'number of pictures generated')
  argparser.add_argument('--paste_image_count_value', nargs='*', type=int, default=[0, 1, 2], help = 'each value for numbers of logos pasted on background image')
  argparser.add_argument('--paste_image_count_probability', nargs='*', type=float, default=[0.1, 0.7, 0.2], help = 'probability of each paste_image_count value')
  argparser.add_argument('--repeat_value', nargs='*', type=int, default=[1, 2, 3], help = 'each value for repeat count of each logo')
  argparser.add_argument('--repeat_probability', nargs='*', type=float, default=[0.7, 0.2, 0.1], help = 'probability of each repeat value')
  argparser.add_argument('--is_GaussianBlur_probability', nargs='*', type=float, default=[0.1, 0.9], help = 'probability of whether to do GaussianBlur')
  argparser.add_argument('--radius_value', nargs='*', type=int, default=[1, 2], help = 'each value for radius of GaussianBlur')
  argparser.add_argument('--radius_probability', nargs='*', type=float, default=[1, 0], help = 'probability of each radius value')
  argparser.add_argument('--transparency_rate_value', nargs='*', type=float, default=[1, 0.8, 0.6], help = 'each value for transparency_rate for transparency process')
  argparser.add_argument('--transparency_rate_probability', nargs='*', type=float, default=[0.5, 0.3, 0.2], help = 'probability of each transparency_rate value')
  argparser.add_argument('--aspect_ratio_rate_value', nargs='*', type=float, default=[1, 0.8, 1.2], help = 'each value for aspect_ratio_rate for aspect_ratio process')
  argparser.add_argument('--aspect_ratio_rate_probability', nargs='*', type=float, default=[0.6, 0.2, 0.2], help = 'probability of each aspect_ratio_rate value')
  argparser.add_argument('--location_probability', nargs='*', type=float, default=[0.2, 0.2, 0.2, 0.2, 0.2], help = 'probability of each loacation to paste')
  argparser.add_argument('--bg_size_limit', nargs='*', type=float, default=[600, 400], help = 'minsize of background image')
  FLAGS, unparsed = argparser.parse_known_args()
  print(FLAGS)
  if len(unparsed) > 0:
    print("there are unknow args %s " % ','.join(unparsed))
    sys.exit(-1)
  background_image_dir = FLAGS.background_image_dir
  paste_image_gt_file = FLAGS.paste_image_gt_file
  save_dir = FLAGS.save_dir
  train_boxes_list = os.path.join(save_dir,'train_boxes.list')
  total = FLAGS.total
  paste_image_count_value = FLAGS.paste_image_count_value
  paste_image_count_probability = FLAGS.paste_image_count_probability
  repeat_value = FLAGS.repeat_value
  repeat_probability = FLAGS.repeat_probability
  is_GaussianBlur_probability = FLAGS.is_GaussianBlur_probability
  radius_value = FLAGS.radius_value
  radius_probability = FLAGS.radius_probability
  transparency_rate_value = FLAGS.transparency_rate_value
  transparency_rate_probability = FLAGS.transparency_rate_probability
  aspect_ratio_rate_value = FLAGS.aspect_ratio_rate_value
  aspect_ratio_rate_probability = FLAGS.aspect_ratio_rate_probability
  location_probability = FLAGS.location_probability
  bg_size_limit = FLAGS.bg_size_limit
  if not os.path.exists(background_image_dir):
    print('background image dir not exists')
    sys.exit(-1)
  if not os.path.exists(paste_image_gt_file):
    print('paste image file not exists')
    sys.exit(-1)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if background_image_dir.endswith('.list') or background_image_dir.endswith('.txt'):
    with open(background_image_dir, 'r') as f:
      background_image_list = [line.strip() for line in f.readlines()]
  else:
    assert os.path.isdir(background_image_dir)
    background_image_list = [os.path.join(background_image_dir, image_file) for image_file in os.listdir(background_image_dir) if os.path.isfile(os.path.join(background_image_dir, image_file))]
  with open(paste_image_gt_file,'r') as f:
    paste_image_gt_list = [line.strip() for line in f.readlines()]
  start_time = time.time()
  while(success < total):
    main_processor(background_image_list, paste_image_gt_list, save_dir)
  with open(train_boxes_list,'a+') as f:
    f.write(Buffer)
  end_time = time.time()
  print('total time:%s' % (end_time - start_time))
