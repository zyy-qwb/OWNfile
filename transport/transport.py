import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
plt.rcParams['font.sans-serif']='SimHei'
data=pd.read_excel("data_wuliu.xlsx")
#删除重复值
data.drop_duplicates(keep='first',inplace=True)
#删除缺失值,带有na的整行数据
data.dropna(axis=0,how='any',inplace=True)
#删除订单行,重置索引
data.drop(columns=['订单行'],inplace=True,axis=1)
data.reset_index(drop=True,inplace=True)

def data_deal(number):
    if number.find('万元')!=-1:
        new_number=float(number[:number.find('万元')].replace(',',''))*10000
        pass
    else:
        new_number=float(number[:number.find('元')].replace(',',''))
        pass
    return new_number
data['销售金额']=data['销售金额'].map(data_deal)
data.describe()
#销售金额==0，删除，数据量小
data=data[data['销售金额']!=0]
data['销售时间']=pd.to_datetime(data['销售时间'])
data['月份']=data['销售时间'].apply(lambda x:x.month)

# data['货品交货状况']=data['货品交货状况'].str.strip()#去除首尾空格
# data1=data.groupby(['月份','货品交货状况']).size().unstack()

# data1['按时交货率']=data1['按时交货']/(data1['按时交货']+data1['晚交货'])




# data['货品交货状况']=data['货品交货状况'].str.strip()#去除首尾空格
# data1=data.groupby(['货品','销售区域','货品交货状况']).size().unstack()

# data1['按时交货率']=data1['按时交货']/(data1['按时交货']+data1['晚交货'])
# print(data1.sort_values(by='按时交货率',ascending=False))

# data1=data.groupby(['月份','销售区域','货品'])['数量'].sum().unstack()
# data1.plot(kind='line')
# plt.show()
# print(data1)

data['货品用户反馈'] = data['货品用户反馈'].str.strip()  #取出首位空格
data1 = data.groupby(['货品','销售区域'])['货品用户反馈'].value_counts().unstack()
data1['拒货率'] = data1['拒货'] /data1.sum(axis=1)  #按行进行求和汇总
data1['返修率'] = data1['返修'] /data1.sum(axis=1)
data1['合格率'] = data1['质量合格'] /data1.sum(axis=1)
data1.sort_values(['合格率','返修率','拒货率'],ascending=False)
print(data1)