#coding=utf-8
import time
from bs4 import BeautifulSoup
import csv

class save():
    def __init__(self,fname=None):
        if fname:
            self.fname=fname
        else:
            self.fname = './result/loanlist-%s.csv' % str(time.time())

    def paser_bid(self,list_result):
        #提取内容
        soup = BeautifulSoup(list_result, 'html.parser')
        infos=soup.loaninfos
        loanlist=[]
        for inf in infos.children:
            loan = {}
            for child in inf.children:
                if child.string:
                    loan[child.name]=child.string.encode('utf-8')
                else:
                    loan[child.name] = child.string
            loanlist.append(loan)
            # print (loan)
        return loanlist

    def paser_info(self,list_result):
        soup = BeautifulSoup(list_result, 'html.parser')
        bids = 0.0
        for bid in soup.bids.children:
            bids = bids + float(bid.bidamount.string)
        return bids

    def paser_bding(self,bidg):
        soup = BeautifulSoup(bidg, 'html.parser')
        participation=soup.participationamount.string
        return participation



    def saveresult(self,list_result):
        #存储内容
        headers=list(list_result[0].keys())
        with open(self.fname, 'w') as f:
            f_csv = csv.DictWriter(f, headers)
            f_csv.writeheader()
            f_csv.writerows(list_result)
        return


