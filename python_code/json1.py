import json
data = {
    "Kareem Abdul-Jabbar":'kareem@jizhi.im',
    "Karl Malone":'karl@jizhi.im',
    "Kobe Bryant":'kobe@jizhi.im',
    "Michael Jordan":'michael@jizhi.im',
    "Wilt Chamberlain":'wilt@jizhi.im',
    "DIRK NOWITZKI":'dirk@jizhi.im',
    "Shaquille O'Neal":'shaquille@jizhi.im',
    "Moses Malone":'moses@jizhi.im',
    "Elvin Hayes":'elvin@jizhi.im',
    "LeBRON JAMES":'lebron@jizhi.im'
}
with open ('address.json','w') as f:
    json.dump(data ,f)#写入的是文件，dumps把数据类型转化为字符串
with open('address.json', 'r') as f:
    data = json.load(f)#读取的是json文件
json_str = json.dumps(data)
print(type(json_str ))
#data = json.loads(json_str )#读取的是json内存对象
print(json_str)
print(data)