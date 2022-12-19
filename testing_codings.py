from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.header import Header

#邮箱服务器地址，这里我们用的时qq的。要换成163的话这里需要更换。并且如果换成163的话端口号也不一样
mail_host = "smtp.qq.com"
#邮箱登录名
mail_user = '258546048@qq.com'
#密码(部分邮箱为授权码)
mail_pass = 'jifsasoyrgszcbdb'
#邮件发送方邮箱地址
sender = '258546048@qq.com'
#接收邮箱的地址
receiver = '381205977@qq.com'

message = MIMEText('张师兄代码跑完啦', 'plain', 'utf-8')
#邮件主题
message['Subject'] = Header('张师兄代码跑完啦', 'utf-8')
#发送方信息
message['From'] = Header("服务器3090", 'utf-8')
#接受方信息
message['To'] = Header("张师兄", 'utf-8')
import os
import time
def autohalt():
    while True:
        ps_string_1 = os.popen('ps ax | grep 4010988','r').read() # 这里的6666是进程号，后面简单说一下怎么查询
        ps_strings_1 = ps_string_1.strip().split('\n')
        print(ps_strings_1)
        if len(ps_strings_1)<=2:
            message = MIMEText('张师兄代码跑完啦')
            smtp = SMTP_SSL(mail_host)
            smtp.login(mail_user, mail_pass)
            smtp.sendmail(sender, receiver, message.as_string())
            smtp.quit()
            print('success')
            print(message.as_string())
            return
        else:
            print('Still',len(ps_strings_1),'Processes, waiting 5s...')
            time.sleep(5) #一分钟后检查一次
if __name__=='__main__':
    autohalt()
