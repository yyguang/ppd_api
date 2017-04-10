#coding=utf-8
from openapi_client import openapi_client as client
from core.rsa_client import rsa_client as rsa
import pickle
import json
import datetime
import os
import time
from savedata import save
from bs4 import BeautifulSoup


class Dapi():
  def __init__(self):
      self.appid="410a9843cf9444249b02e5cac2de9b95"
      # self.code ="17ef84b82a674697a20bbc8fde746a7b"

  def authorize(self):
      # 授权
      authorizeStr = client.authorize(appid=self.appid,code=self.code) #获得授权
      self.authorizeObj = pickle.loads(authorizeStr)
      # 将返回的authorize对象反序列化成对象，成功得到 OpenID、AccessToken、RefreshToken、ExpiresIn
      # authorize_data:{"OpenID":"8d50843818bf4ef0b91a308b90e9ef87","AccessToken":"ba0a0c70-6463-44d5-8f0c-9ba05a4a2ff5","RefreshToken":"9e5486b8-1087-47b2-ae1a-36404f802c03","ExpiresIn":604800}
  def refresh(self):
      # 刷新令牌
      openid=self.authorizeObj['OpenID']
      refreshtoken=self.authorizeObj['RefreshToken']
      new_token_info = client.refresh_token(self.appid, openid, refreshtoken)


  def BatchLenderBidList(self):
      #（跟投）用户最近投资标的信息（批量）
      access_url = "http://gw.open.ppdai.com/invest/BidService/BatchLenderBidList"
      access_token = "ba0a0c70-6463-44d5-8f0c-9ba05a4a2ff5"
      data = {
        "LenderNames": [
          "fell_2015"
        ],
        "TopIndex": 10
      }
      sort_data = rsa.sort(data)
      sign = rsa.sign(sort_data)
      list_result = client.send(access_url,json.dumps(data) , self.appid, sign,access_token)
      return list_result

  def LoanList(self):
      # 新版投标列表接口（默认每页2000条）
      access_url = "http://gw.open.ppdai.com/invest/LLoanInfoService/LoanList"
      data = {
        "PageIndex": 1,
        # "StartDateTime": "2016-3-1 12:00:00.000"
      }
      sort_data = rsa.sort(data)
      sign = rsa.sign(sort_data)
      list_result = client.send(access_url, json.dumps(data), self.appid, sign)
      return list_result

  def BatchListingInfos(self,loan_ids):
      #新版散标详情批量接口（请求列表不大于10）
      access_url = "http://gw.open.ppdai.com/invest/LLoanInfoService/BatchListingInfos"
      data = {
        "ListingIds": [loan_ids]
      }
      sort_data = rsa.sort(data)
      sign = rsa.sign(sort_data)
      list_result = client.send(access_url,json.dumps(data) , self.appid, sign)
      return list_result

  def BatchListingBidInfos(self,loan_ids):
      # 新版列表投标详情批量接口（请求列表大小不大于5）
      access_url = "http://gw.open.ppdai.com/invest/LLoanInfoService/BatchListingBidInfos"
      data = {
          "ListingIds": [
              loan_ids
          ]
      }
      sort_data = rsa.sort(data)
      sign = rsa.sign(sort_data)
      list_result = client.send(access_url, json.dumps(data), self.appid, sign)
      return list_result
  def biding(self,loan_id,amount):
      # 投标接口
      access_url = "http://gw.open.ppdai.com/invest/BidService/Bidding"
      access_token = "ba0a0c70-6463-44d5-8f0c-9ba05a4a2ff5"
      data = {
          "ListingId": loan_id,
          "Amount": amount
      }
      sort_data = rsa.sort(data)
      sign = rsa.sign(sort_data)
      list_result = client.send(access_url, json.dumps(data), self.appid, sign, access_token)
      return list_result

class deal_bid():
    def __init__(self):
        self.api = Dapi()
        list_result = self.api.LoanList()
        filename = './result/BatchListingInfos-%s.csv' % str(time.time())
        self.sv = save(filename)
        self.loan_list = self.sv.paser_bid(list_result)

    def rmfunding(self):
        # 获取抓取标的可投资金额，即可投资总额减已投资金额
        for loan in self.loan_list:
            id = loan['listingid']
            result = self.api.BatchListingBidInfos(id)
            loan_sum = self.sv.paser_info(result)
            remainfunding = float(loan['amount']) - loan_sum
        return remainfunding

    def binfos(self):
        # 获取抓取标的详细信息，以字典形式拼成列表返回
        bid_infos = []
        for loan in self.loan_list:
            id = loan['listingid']
            result = self.api.BatchListingInfos(id)
            bid_info = self.sv.paser_bid(result)
            print (bid_info)
            bid_infos.append(bid_info[0])
        # self.sv.saveresult(bid_infos)
        return bid_infos

class BidMount():
    def __init__(self,levmount):
        self.funding = [levmount['mount1'],levmount['mount2'],levmount['mount3'],levmount['mount4'],levmount['mount5']]
        self.api = Dapi()
        self.sv = save()
        bidlist = deal_bid()
        self.bid_infos = bidlist.binfos()
        # 将获取的标按风险分成五个等级
        ption_num=len(self.bid_infos)/5
        bid_infos1=self.bid_infos[0:ption_num]
        bid_infos2=self.bid_infos[ption_num:2*ption_num]
        bid_infos3 =self.bid_infos[2*ption_num:4*ption_num]
        bid_infos4 =self.bid_infos[4*ption_num:5*ption_num]
        bid_infos5 =self.bid_infos[5*ption_num:]
        self.bid_info_list=[bid_infos1,bid_infos2,bid_infos3,bid_infos4,bid_infos5]
        # # self.levids=levmodel(bid_infos)
        # for bid in self.bid_info_list:
        #     print (bid)

    def bidmount(self):
        # 给定5个等级的标投资金额（levmount），根据模型输出的对应5个等级的标(levmodel)进行投标
        level=['less than 5%','5%-10%','10%-15%','15%-20%','higher than 20%']
        for i in range(5):
            fund=self.funding[i]
            if fund !=0:
                for bid in self.bid_info_list[i]:
                    bidg = self.api.biding(bid['listingid'], fund)
                    participation = self.sv.paser_bding(bidg)
                    fund = int(fund) - int(participation)
                    print (level[i],bid['listingid'], participation, fund)
                    if fund == 0:
                        break
        return
        # return sum(fund)


if __name__=='__main__':
    input={'mount1':3200,'mount2':1200,'mount3':800,'mount4':600,'mount5':500}
    bid=BidMount(input)
    totalfund=bid.bidmount()
    # 此处采用递归，直至所有金额投资完成
    # if totalfund<sum(input):
    #     __main__()
    # else:
    #     pass




# filename = './result/LoanList-%s.csv' % str(time.time())
# sv=save(filename)
# sv.saveresult(list_result)