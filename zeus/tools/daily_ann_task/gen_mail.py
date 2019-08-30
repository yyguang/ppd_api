# -*- coding: UTF-8 -*-
import sys
import time
import os
path = sys.argv[1]
task_type = sys.argv[2]
total_count = 0

key_dict = {'shumei_dianshang':'数美电商广告任务','shumei_shouxiedazi':'数美手写大字广告任务','shumei_shouxieti':'数美手写体广告任务','shumei_weixin':'数美微信广告任务','shumei_jupaizi':'数美举牌子广告任务','shumei_lianxifangshishuiyin':'数美联系方式广告任务','shumei_zhengchang':'数美正常广告任务',\
            'duiduikeji_normal': '对对科技正常任务','duiduikeji_sexy': '对对科技性感任务','duiduikeji_porn': '对对科技色情任务','tongzhuo_normal': '同桌正常任务','tongzhuo_sexy': '同桌性感任务','tongzhuo_porn': '同桌色情任务', 'vipkid': 'vipkid任务', 'xiaohongshu': '小红书任务', 'xiongmaotv': '熊猫TV任务', 'rela': '热拉', 'xiaoying': '小影', 'uki': 'Uki', 'yizhibo': '一直播', 'huanqiubushou': '环球捕手', 'maimai_normal': '脉脉正常任务', 'maimai_sexy': '脉脉性感任务', 'maimai_porn': '脉脉色情任务', 
            'shumei_smoke':"数美吸烟任务",'tupu_smoke':"图普吸烟任务",\
			'shumei_politics':"数美涉政人物任务",\
            'shumei_anheidongman':"数美暗黑动漫任务",'shumei_ertongxiedian':"数美儿童邪典任务", 'shumei_kongbuzuzhi':"数美恐怖组织任务",'shumei_baoluanchangjing':'数美暴乱场景任务','shumei_guoqiguohui':'数美国旗国徽任务', 'shumei_junzhuang':'数美军装任务','shumei_qiangzhidaoju':'数美枪支刀具任务','shumei_xuexingchangjing':'数美血腥场景任务', 'tupu_terror':"图普暴恐任务",'shumei_zhengchangzongjiao':'数美正常宗教任务'
            }
keys = {"ad":['shumei_dianshang','shumei_shouxieti','shumei_shouxiedazi','shumei_weixin','shumei_lianxifangshishuiyin','shumei_jupaizi'],\
        "porn":['duiduikeji_normal','duiduikeji_sexy','duiduikeji_porn','tongzhuo_normal','tongzhuo_sexy','tongzhuo_porn', 'vipkid', 'xiaohongshu', 'xiongmaotv', 'rela', 'xiaoying', 'uki', 'yizhibo', 'huanqiubushou', 'maimai_normal', 'maimai_sexy', 'maimai_porn'],\
        "violence":['shumei_ertongxiedian','shumei_baoluanchangjing','shumei_anheidongman','shumei_xuexingchangjing'] ,\
        "smoke" :['shumei_smoke'],\
	"politics":['shumei_politics']
        }

key_list = keys[task_type]
for i in key_list:
  #shumei_dianshang_task.list
  text = path + '/' + i + '_task_list.list'
  list = path + '/' + i + '_task.list'

  line = 'cat' + ' ' + list + '|wc -l'

  count = int(os.popen(line).read())
  if count > 0:
    total_count += count
    with open(text,'r') as f:
      a = f.read()
      print key_dict[i], a,"\t总量:",count

print "发送时间:",time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))
print "任务总量:",total_count
print "优先级:T1"

