"""
下面的文件将会从csv文件中读取读取短信与电话记录，
你将在以后的课程中了解更多有关读取文件的知识。
"""
import csv

with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)

"""
任务2: 哪个电话号码的通话总时间最长? 不要忘记，用于接听电话的时间也是通话时间的一部分。
输出信息:
"<telephone number> spent the longest time, <total time> seconds, on the phone during
September 2016.".

提示: 建立一个字典，并以电话号码为键，通话总时长为值。
这有利于你编写一个以键值对为输入，并修改字典的函数。
如果键已经存在于字典内，为键所对应的值加上对应数值；
如果键不存在于字典内，将此键加入字典，并将它的值设为给定值。
"""
new_list=list()
new_dict={}
new_set=set()

#创建一个集合，将电话号码放进去
for i,rows1 in enumerate(calls):
    new_set.add(rows1[0])
    new_set.add(rows1[1])

#作为字典添加
for j in new_set:
    new_dict.setdefault(j,0)
    
#两重遍历，同样号码的，进行通话时间叠加
for key,value in new_dict.items():
    for rows1 in calls:
        if key==rows1[0] or key==rows1[1]:
            new_dict[key]=new_dict[key]+int(rows1[3])
            
new_list=sorted(new_dict.items(),key=lambda d: d[1])
print(str(new_list[-1][0])+" spent the longest time,"+ str(new_list[-1][1]) +" seconds, on the phone during September 2016.")

            
        