import json, re, codecs

emails = {}

def trash_mail(mail_list, mail_template): # 定义函数
    with open(mail_list, 'r') as f:
        dict_addr = json.load(f)
    with open (mail_template ,'r') as f:
        f_temp = f.read()
    for i in dict_addr: # 对收件人列表循环
        to_replace = re.sub('\[address]', dict_addr[i], f_temp) # 替换[address]为收件人地址
        to_replace = re.sub('\[name]',i,to_replace)
        emails[i] = to_replace
mail_list = "address.json" # 邮件群发地址的json文件
mail_template = "emial.txt" # 邮件模板txt文件
trash_mail(mail_list, mail_template) # 执行函数，传入参数
# 查看邮件内容（重点是[address]和[name]是否替换成功）
print(emails['Kobe Bryant'])