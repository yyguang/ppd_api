#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author: chenyunkuo2018-12-12 14:19:18
# Intro: Create data augumentation images from case.

import os
import cv2
import sys
import glob
import random
import shutil
from PIL import Image, ImageEnhance

def md5_rename(img_path):
  md5name = os.popen("md5sum {}".format(img_path)).read().split()[0]
  root_dir = "/".join(img_path.split("/")[: -1])
  os.system("mv {} {}".format(img_path, os.path.join(root_dir, md5name+".jpg")))

def check_extension(img_path):
  return True if img_path.split(".")[-1] in {'jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG'} else False

# 色彩变换
def color_aug(img_path, output):
  image = Image.open(img_path)
  # 亮度增强
  enh_bri = ImageEnhance.Brightness(image)
  brightness = 1.5
  image_brightened = enh_bri.enhance(brightness)
  # 色度增强
  enh_col = ImageEnhance.Color(image)
  color = 1.5
  image_colored = enh_col.enhance(color)
  # 对比度增强
  enh_con = ImageEnhance.Contrast(image)
  contrast = 1.5
  image_contrasted = enh_con.enhance(contrast)
  # 锐度增强
  enh_sha = ImageEnhance.Sharpness(image)
  sharpness = 3.0
  image_sharped = enh_sha.enhance(sharpness)
  
  basename = os.path.basename(img_path).split(".")[0]
  image_brightened.save(os.path.join(output, basename + "_brightened.jpg"))
  image_colored.save(os.path.join(output, basename + "_colored.jpg"))
  image_contrasted.save(os.path.join(output, basename + "_contrasted.jpg"))
  image_sharped.save(os.path.join(output, basename + "_sharped.jpg"))

# 几何变换:翻转、镜像
def geom_aug(img_path, output):
  image = Image.open(img_path)
  # 翻转
  image_rotate90 = image.rotate(90)
  image_rotate180 = image.rotate(180)
  image_rotate270 = image.rotate(270)
  # 镜像
  image_hflip = image.transpose(Image.FLIP_LEFT_RIGHT)
  image_vflip = image.transpose(Image.FLIP_TOP_BOTTOM)
  
  basename = os.path.basename(img_path).split(".")[0]
  image_rotate90.save(os.path.join(output, basename + "_rotate90.jpg"))
  image_rotate180.save(os.path.join(output, basename + "_rotate180.jpg"))
  image_rotate270.save(os.path.join(output, basename + "_rotate270.jpg"))
  image_hflip.save(os.path.join(output, basename + "_hflip.jpg"))
  image_vflip.save(os.path.join(output, basename + "_vflip.jpg"))
  
# 风格化变换
def cartoonise_aug(img_path, output):
  img_rgb = cv2.imread(img_path)
  num_bilateral = 7    # 定义双边滤波的数目
  img_color = img_rgb
  for _ in range(num_bilateral):
    img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
  img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
  img_blur = cv2.medianBlur(img_gray, 15) # 调大该值可以减少噪点，必须为奇数
  # 检测到边缘并且增强其效果
  img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   blockSize=7, # 控制轮廓的粗细
                                   C=4) # 细节丰富程度
  # 转换回彩色图像
  img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
  img_cartoon = cv2.bitwise_and(img_color, img_edge)
  # 保存转换后的图片
  basename = os.path.basename(img_path).split(".")[0]
  cv2.imwrite(os.path.join(output, basename + "_cartoon.jpg"), img_cartoon)

# 随机裁剪
def crop_aug(img_path, output):
  image = Image.open(img_path)
  w, h = image.size
  basename = os.path.basename(img_path).split(".")[0]
  # 边缘裁剪2像素
  image_peeled = image.crop([2, 2, w - 2, h - 2])
  # random crop
  for w_ratio in [0.8, 0.9]:
    for h_ratio in [0.8, 0.9]:
      w_new = int(round(w * w_ratio))
      h_new = int(round(h * h_ratio))
      x_start = random.randint(0, w - w_new)
      y_start = random.randint(0, h - h_new)
      image_cropped = image.crop([x_start, y_start, x_start + w_new, y_start + h_new])
      image_cropped.save(os.path.join(output, basename + "_{}x{}.jpg".format(w_new, h_new)))

if __name__ == '__main__':
  input = sys.argv[1]
  output = sys.argv[2]

  imgs = []
  if os.path.isdir(input):
    for p in glob.glob(os.path.join(input, "*", "*")):
      if check_extension(p):
        imgs.append(p)
    for p in glob.glob(os.path.join(input, "*")):
      if check_extension(p):
        imgs.append(p)
  elif os.path.exists(input):
    if check_extension(input):
      imgs.append(input)
  else:
    print("Input should be dir with images or an image path.")
  if os.path.isdir(output):
    shutil.rmtree(output)
  os.mkdir(output)
 
  for img_path in imgs:
    color_aug(img_path, output)
    geom_aug(img_path, output)
    cartoonise_aug(img_path, output)
    crop_aug(img_path, output)
  for p in glob.glob(os.path.join(output, "*")):
    md5_rename(p)
  print("Augumentation finished.")
