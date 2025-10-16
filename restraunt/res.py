import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']='SimHei'
data1=pd.read_excel('meal_order_detail.xlsx',sheet_name='meal_order_detail1')
data2=pd.read_excel('meal_order_detail.xlsx',sheet_name='meal_order_detail2')
data3=pd.read_excel('meal_order_detail.xlsx',sheet_name='meal_order_detail3')
data=pd.concat([data1,data2,data3],axis=0)
# print(data.info())
data.dropna(axis=1,inplace=True)
# print(data.info())
round(np.mean(data['amounts']),2)
dishes_count=data['dishes_name'].value_counts()[:10]
# print(dishes_count)

# dishes_count.plot(kind="bar",fontsize=12)
# dishes_count.plot(kind='line',color=['r'])

# for x,y in enumerate(dishes_count):
#     plt.text(x,y+2,y,ha='center',fontsize=12)

# plt.xticks(rotation=45)
# plt.show()

# data_group=data['order_id'].value_counts()[:10]
# data_group.plot(kind="bar",fontsize=12)
# plt.title("订单点菜种类top10")
# plt.xlabel("订单号")
# plt.ylabel("点菜种类")
# plt.show()

# data['total_amounts']=data['counts']*data['amounts']
# dataGroup=data[['order_id','counts','amounts','total_amounts']].groupby(by="order_id")
# Group_sum=dataGroup.sum()
# sort_counts=Group_sum.sort_values(by="counts",ascending=False)
# sort_counts['counts'][:10].plot(kind="bar",fontsize=12)
# plt.title("订单点菜数量top10")
# plt.xlabel("订单号")
# plt.ylabel("点菜数量")
# plt.show()

# sort_counts=Group_sum.sort_values(by="total_amounts",ascending=False)
# sort_counts['total_amounts'][:10].plot(kind="bar",fontsize=12)
# plt.title("订单消费top10")
# plt.xlabel("订单号")
# plt.ylabel("消费金额")
# plt.show()

# Group_sum["average"]=Group_sum['total_amounts']/Group_sum['counts']
# sort_average=Group_sum.sort_values(by='average',ascending=False)
# sort_average['average'][:10].plot(kind="bar")

data['hourcount']=1
data['time']=pd.to_datetime(data['place_order_time'])
data['hour']=data['time'].map(lambda x:x.hour)
gp_byhour=data.groupby(by='hour').count()['hourcount']
print(gp_byhour)
