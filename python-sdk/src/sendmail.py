
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import smtplib, datetime

def sendmail(send_account, accept_account, result_csv):
    #创建一个带附件的实例
    msg = MIMEMultipart()

    #加邮件头
    msg['to'] = ';'.join(accept_account)
    msg['from'] = send_account
    msg['subject'] = Header('撞库结果推送('+ str(datetime.date.today())+ ')','utf-8')

    #添加邮件正文
    mail_body='附件是撞库结果，请查收！'
    body=MIMEText(mail_body, 'plain', 'gb2312')
    msg.attach(body)

    #构造附件
    with open(result_csv,'rb') as f:
        att = MIMEText(f.read(),'base64','gb2312')
    att["Content-Type"] = 'application/octet-stream'
    att["Content-Disposition"] = 'attachment; filename=result.csv'
    msg.attach(att)

    #发送邮件
    smtp=smtplib.SMTP()
    smtp.connect('smtp.zju.edu.cn')
    smtp.login('guangyang@zju.edu.cn','081256')
    smtp.sendmail(send_account, accept_account,msg.as_string())
    smtp.quit()

if __name__ == '__main__':
    sendmail('guangyang@zju.edu.cn', ['vilon130@163.com','guangyang@zju.edu.cn'], './log/hy_website.csv')
