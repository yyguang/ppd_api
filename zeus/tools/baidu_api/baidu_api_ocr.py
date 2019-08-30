#!/usr/bin/env python
#-*- coding:utf8 -*-
# Author Tony Tsao <teng.cao@foxmail.com> 2018-01-08 14:27:34

from aip import AipImageCensor, AipOcr
import json
import sys


'''
def get_file_content(filePath):
    with open(filePath, 'r') as fp:
        return fp.read()

client = AipImageCensor("10637861", "SrkhZ3xliBiXnreoRa1lBTbM", "EMFiYsmOuGKb0A9buXuFjbBDMpKE2uoC")

result = client.imageCensorUserDefined(get_file_content(sys.argv[1]))
if result["conclusion"] == u"不合规":
    datas=result['data']
    for i in range(len(datas)):
        print(datas[i]['msg'])

print result
'''

'''
""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

client = AipOcr("10637861", "SrkhZ3xliBiXnreoRa1lBTbM", "EMFiYsmOuGKb0A9buXuFjbBDMpKE2uoC")
image = get_file_content(sys.argv[1])

""" 调用通用文字识别（含位置高精度版） """
client.accurate(image);

""" 如果有可选参数 """
options = {}
options["recognize_granularity"] = "big"
options["detect_direction"] = "true"
options["vertexes_location"] = "true"
options["probability"] = "true"

""" 带参数调用通用文字识别（含位置高精度版） """
result = client.accurate(image, options)
print result
'''

""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

client = AipOcr("10637861", "SrkhZ3xliBiXnreoRa1lBTbM", "EMFiYsmOuGKb0A9buXuFjbBDMpKE2uoC")

image = get_file_content(sys.argv[1])

""" 调用通用文字识别（含位置信息版）, 图片参数为本地图片 """
client.general(image);

""" 如果有可选参数 """
options = {}
options["recognize_granularity"] = "big"
options["language_type"] = "CHN_ENG"
options["detect_direction"] = "true"
options["detect_language"] = "true"
options["vertexes_location"] = "true"
options["probability"] = "true"

""" 带参数调用通用文字识别（含位置信息版）, 图片参数为本地图片 """
result = client.general(image, options)
print result

