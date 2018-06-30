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
任务4:
电话公司希望辨认出可能正在用于进行电话推销的电话号码。
找出所有可能的电话推销员:
这样的电话总是向其他人拨出电话，
但从来不发短信、接收短信或是收到来电


请输出如下内容
"These numbers could be telemarketers: "
<list of numbers>
电话号码不能重复，每行打印一条，按字典顺序排序后输出。
"""
new_dict=dict()
new_set=set()

result=set()
remove=set()


for i,rows3 in enumerate(calls):
    #取出所有的呼叫电话
    result.add(rows3[0])
    #取出所有被呼叫电话
    remove.add(rows3[1])
    
for i,rows5 in enumerate(texts):
    #取出发短信的电话
    remove.add(rows5[0])
    #取出收短信的电话
    remove.add(rows5[1])
    
for i,rows4 in enumerate(result):
    if rows4 not in remove :
        new_set.add(rows4)

# print(len(new_set))
# for j,rows2 in enumerate(new_set):
#     new_dict[j]=rows2
#  
# #根据字典顺序排序,返回是一个list
# new_list=sorted(new_dict.items(),key=lambda item:item[1])
#  
# print("These numbers could be telemarketers: ")
# print(len(new_list))
# for m in new_list:
#     print(m[1])
    
print("These numbers could be telemarketers: "+ "\n" + "\n".join(sorted(new_set)))
