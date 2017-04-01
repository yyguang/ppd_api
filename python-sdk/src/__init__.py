#coding=utf-8
from openapi_client import openapi_client as client
from core.rsa_client import rsa_client as rsa
import pickle
import json
import time
import datetime
import os

appid="410a9843cf9444249b02e5cac2de9b95"
# code = ""
#
# #step 1 授权
# authorizeStr = client.authorize(appid=appid,code=code) #获得授权
# authorizeObj = pickle.loads(authorizeStr) # 将返回的authorize对象反序列化成对象，成功得到 OpenID、AccessToken、RefreshToken、ExpiresIn
# #{"OpenID":"xx","AccessToken":"xxx","RefreshToken":"xxx","ExpiresIn":604800}
#
# #step 1 刷新令牌
# openid=""
# refreshtoken= ""
# new_token_info = client.refresh_token(appid, openid, refreshtoken)
#
# #step 2 发送数据（可投标列表接口）


access_url = "http://gw.open.ppdai.com/auth/registerservice/register"
utctime = datetime.datetime.utcnow()
data = {
  "Mobile": "15158058779",
  "Email": "1109396480@qq.com",
  "Role": 12
}
sort_data = rsa.sort(data)
sign = rsa.sign(sort_data)
list_result = client.send(access_url,json.dumps(data) , appid, sign)


