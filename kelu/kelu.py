import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
from datetime import datetime

df=pd.read_csv('kelu.csv')
df.info()
df.head()
df.describe()
print(df.describe())
df['time'] = pd.to_datetime(df['time'],format='%Y-%m-%d')
# df.groupby('time')['rating'].count().plot(figsize=(12,4))
# plt.show()

df['month']=df['time'].values.astype('datetime64[M]')

# df.groupby('month')['rating'].count().plot(figsize=(12,4))
# plt.xlabel('月份')
# plt.ylabel('销售数据')
# plt.title('16-19年每月销量分析')
# plt.show()


#按照游客分组,统计每个游客的购买次数
grouped_count_author=df.groupby('author')['frequency'].count().reset_index()
#统计每个游客的消费金额
grouped_sum_amount=df.groupby('author')['amount'].sum().reset_index()
user_purchase_retention=pd.merge(left=grouped_count_author,
                                right=grouped_sum_amount,
                                on='author',
                                how='inner')
# print(user_purchase_retention.tail(60))
# user_purchase_retention.plot.scatter(x='frequency',y='amount',figsize=(12,4))
# plt.title('用户的购买次数和消费金额关系图')
# plt.xlabel('购买次数')
# plt.ylabel('消费金额')
# plt.show()

# df.groupby('author')['frequency'].count().plot.hist(bins=50)
# plt.xlim(1,17)
# plt.xlabel('购买数量')
# plt.ylabel('人数')
# plt.title('用户购买门票数量直方图')
# plt.show()

df_frequency_2=df.groupby('author').count().reset_index()
# df_frequency_2.head()
# df_frequency_2[df_frequency_2['frequency']>=2].groupby('author')['frequency'].sum().plot.hist(bins=50)
# plt.xlabel('购买数量')
# plt.ylabel('人数')
# plt.title('购买门票在2次及以上的用户数量')
# plt.show()

# print(df_frequency_2[df_frequency_2['frequency']>=2].groupby('frequency')['author'].count())
df_frequency_gte_1=df.groupby('author')['frequency'].count().reset_index()

values=list(df_frequency_gte_1[df_frequency_gte_1['frequency']<=5].groupby('frequency')['frequency'].count())
# print(values)
# plt.pie(values,labels=['购买一次','购买两次','购买三次','购买四次','购买五次'],autopct='%1.1f%%')
# plt.title('购买次数在1-5次之间的人数占比')
# plt.legend()
# plt.show()

# df_frequency_gte_2=df_frequency_2[df_frequency_2['frequency']>=2].reset_index()

# values=list(df_frequency_gte_2[df_frequency_gte_2['frequency']<=5].groupby('frequency')['frequency'].count())
# print(values)
# plt.pie(values,labels=['购买两次','购买三次','购买四次','购买五次'],autopct='%1.1f%%')

# plt.title('购买次数在2-5次之间的人数占比')
# plt.legend()
# plt.show()

pivot_count=df.pivot_table(index='author',
                           columns='month',
                           values='frequency',
                           aggfunc='count').fillna(0)
print(pivot_count.head())
# pivot_count=pivot_count.applymap(lambda x:1 if x>1 else np.nam if x==0 else 0)
# (pivot_count.sum()/pivot_count.count()).plot()
# plt.xlabel('时间（月）')
# plt.ylabel('百分比(%)')
# plt.title('16-19年每月用户复购率')
# plt.show()

# pivot_count.sum().plot()
# plt.xlabel('时间(月)')
# plt.ylabel('复购人数')
# plt.title('16-19年每月复购人数折线图')
# plt.show()

pivot_purchase=df.pivot_table(index='author',
                           columns='month',
                           values='frequency',
                           aggfunc='count').fillna(0)
pivot_purchase.head()
len(pivot_purchase.columns)
def purchase_return(data):
    status=[]
    for i in range(30):
        if data[i]==1:
            if data[i+1]==1:
                status.append(1)
            else:
                status.append(0)
        else:
            status.append(np.nan)
    status.append(np.nan)
    return pd.Series(status,pivot_purchase.columns)
pivot_purchase_return=pivot_purchase.apply(purchase_return,axis=1)
# (pivot_purchase_return.sum()/pivot_purchase_return.count()).plot()
# plt.title('16-19年每月回购率')
# plt.xlabel('月份')
# plt.ylabel('回购率%')
# plt.show()

# pivot_purchase_return.sum().plot()
# plt.title('16-19年每月回购人数')
# plt.xlabel('月份')
# plt.ylabel('回购人数')
plt.show()

def active_status(data):#data:每一行数据（18列）
    status=[]
    for i in range(31):
        if data[i]==0:#判断本月消费是否为0
            if len(status)==0:#前几个月没有任何消费记录
                status.append('unreg')
            else:
                if status[i-1]=='unreg':
                    status.append('unreg')
                else:
                    status.append("unactive")
        else:
            if len(status)==0:
                status.append('new')
            else:
                if status[i-1]=='unactive':
                    status.append('return')
                elif status[i-1]=='unreg':
                    status.append('new')
                else:
                    status.append('active')
    return pd.Series(status,pivot_purchase.columns)
pivot_purchase_status=pivot_purchase.apply(active_status,axis=1)
pivot_status_count=pivot_purchase_status.replace('unreg',np.nan).apply(pd.value_counts)
pivot_status_count.T.plot.area()
return_rate=pivot_status_count.apply(lambda x:x/x.sum())
return_rate.T.plot()
return_rate.T['active'].plot(figsize=(12,6))
plt.xlabel('时间/月')
plt.ylabel('百分比')
plt.title('每月活跃用户百分比分析')
plt.show()

return_rate.T['return'].plot(figsize=(12,6))
plt.xlabel('时间/月')
plt.ylabel('百分比')
plt.title('每月回流用户百分比分析')
plt.show()
print(np.mean(return_rate.T['return']))

#用户的生命周期
time_min=df.groupby('author')['time'].min()
time_max=df.groupby('author')['time'].max()
life_time=(time_max-time_min).reset_index()
print(life_time.describe)

#直方图
life_time['life_time']=life_time['time']/np.timedelta64(1,'D')#把日期格式转换为天的精度的数据
life_time['life_time'].plot.hist(bins=100,figsize=(12,6))
plt.xlabel('天数')
plt.ylabel('人数')
plt.title('所有用户的生命周期直方图')
print(life_time[life_time['life_time']==0])

#生命周期大于0的用户直方图
life_time[life_time['life_time']>0]['life_time'].plot.hist(bins=100,figsize=(12,6))
plt.xlabel('天数')
plt.ylabel('人数')
plt.title('生命周期大于0的用户直方图')
plt.show()
print(life_time[life_time['life_time']>0]['life_time'].mean())

#各时间段的用户留存率
np.random.seed(666)
score_list=np.random.randint(25,100,size=3)
print(score_list)
bins=[0,59,70,80]
score_cut=pd.cut(score_list,bins)

user_purchase_retention=pd.merge(left=df,right=time_min.reset_index(),how='inner',on='author',suffixes=('','_min'))
user_purchase_retention['time_diff']=user_purchase_retention['time']-user_purchase_retention['time_min']
user_purchase_retention['time_diff']=user_purchase_retention['time_diff'].apply(lambda x: x/np.timedelta64(1,'D'))

bin=[i*90 for i in range(11)]
user_purchase_retention['time_diff_bin']=pd.cut(user_purchase_retention['time_diff'],bin)

pivot_retention=user_purchase_retention.groupby(['author','time_diff_bin'])['frequency'].sum().unstack()

#判断用户是否留存(1:留存，0：未留存)
pivot_retention_trans=pivot_retention.fillna(0).applymap(lambda x: 1 if x>0 else 0)
print(pivot_retention_trans.sum()/pivot_retention_trans.count())
(pivot_retention_trans.sum()/pivot_retention.count()).plot.bar()
plt.xlabel('时间跨度/天')
plt.ylabel('留存率')
plt.title('各时间段内用户留存率')
plt.show()