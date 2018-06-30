"""
下面的文件将会从csv文件中读取读取短信与电话记录，
你将在以后的课程中了解更多有关读取文件的知识。
"""
import csv
import re

with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)

"""
任务3:
(080)是班加罗尔的固定电话区号。
固定电话号码包含括号，
所以班加罗尔地区的电话号码的格式为(080)xxxxxxx。

第一部分: 找出被班加罗尔地区的固定电话所拨打的所有电话的区号和移动前缀（代号）。
 - 固定电话以括号内的区号开始。区号的长度不定，但总是以 0 打头。
 - 移动电话没有括号，但数字中间添加了
   一个空格，以增加可读性。一个移动电话的移动前缀指的是他的前四个
   数字，并且以7,8或9开头。
 - 电话促销员的号码没有括号或空格 , 但以140开头。

输出信息:
"The numbers called by people in Bangalore have codes:"
 <list of codes>
代号不能重复，每行打印一条，按字典顺序排序后输出。

第二部分: 由班加罗尔固话打往班加罗尔的电话所占比例是多少？
换句话说，所有由（080）开头的号码拨出的通话中，
打往由（080）开头的号码所占的比例是多少？

输出信息:
"<percentage> percent of calls from fixed lines in Bangalore are calls
to other fixed lines in Bangalore."
注意：百分比应包含2位小数。
"""
#第一题
A=list()
B=dict()
new_list=list()
new_set2=set()

#写一个函数作为拆分字符串
def go_split(s, symbol):
    # 拼接正则表达式
    symbol = "[" + symbol + "]+"
    # 一次性分割字符串
    result = re.split(symbol, s)
    # 去除空字符
    return [x for x in result if x]

for a in calls:
    if a[0][:5]=="(080)" :
        A.append(a[1])

new_set1=set(A)

for i,rows1 in enumerate(new_set1):
    if rows1[:2]=="(0":
        new_set2.add((go_split(rows1, "()"))[0])
    if rows1[5]==" ":
        new_set2.add(rows1[:4])
        
    

#转化为字典
for i,rows2 in enumerate(new_set2):
    B[i]=rows2

#根据字典顺序排序,返回是一个list
C=sorted(B.items(),key=lambda item:item[1])


print("The numbers called by people in Bangalore have codes:")
for c in C:
    print(c[1])

#第二题
D=list()
for a in calls:
    if a[0][:5]=="(080)" and a[1][:5]=="(080)" :
        D.append(a[1])
        
persent=len(D)/len(A)*100
print("{} percent of calls from fixed lines in Bangalore are calls to other fixed lines in Bangalore.".format('%.2f%%' % (persent)))