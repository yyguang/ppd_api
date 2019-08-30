import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import urllib.request
import re

#keywords_list = sys.argv[1]
#save_dir = sys.argv[2]
page_num = 2
keynames = ["短裙"]
#with open(keywords_list, 'r') as f:
  #for line_ in f:
    #line = line_.strip()
    #keynames.append(line)
for key in keynames:
  keyname=urllib.request.quote(key)
  headers=('User-Agent',"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36")
  opener=urllib.request.build_opener()
  opener.addheaders=[headers]
  urllib.request.install_opener(opener)
  for i in range(0,page_num):
    url='https://s.taobao.com/search?q='+keyname+'&imgfile=&commend=all&ssid=s5-e&search_type=item&sourceId=tb.index&spm=a21bo.2017.201856-taobao-item.1&ie=utf8&initiative_id=tbindexz_20170306&bcoffset=4&ntoffset=4&p4ppushleft=1%2C48&s='+str(i*44)
    data=urllib.request.urlopen(url).read().decode('utf-8','ignore')
    pat='pic_url":"//(.*?)"'
    imagelist=re.compile(pat).findall(data)
    for j in range(0,len(imagelist)):
      thisimage=imagelist[j]
      thisurl='http://'+thisimage
      print(thisurl)
    #file='~/Download/pics/taobao/'+str(i)+str(j)+'.jpg'
    #urllib.request.urlretrieve(thisurl,file)
