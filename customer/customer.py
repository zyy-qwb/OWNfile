import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
# %matplotlib inline
plt.style.use('ggplot')
plt.rcParams['font.sans-serif']=['SimHei']
columns=['user_id','order_dt','order_products','order_amount']
df=pd.read_table('CDNOW_master.txt',names=columns,sep='\s+')
 #sep:'\s+':匹配任意个空格
df.head()
df.describe()

df['order_date']=pd.to_datetime(df['order_dt'],format='%Y%m%d')
df['month'] = df['order_date'].dt.to_period('M')#astype 粒度不不够

# plt.figure(figsize=(30,15))
# plt.subplot(221)
# df.groupby(by='month')['order_products'].sum().plot()
# plt.title('每月产品购买数量')

# plt.subplot(222)
# df.groupby(by='month')['order_amount'].sum().plot()
# plt.title("每月消费金额")

# plt.subplot(223)
# df.groupby(by='month')['user_id'].count().plot()
# plt.title("每月消费次数")

# plt.subplot(224)
# df.groupby(by='month')['user_id'].apply(lambda x:len(x.drop_duplicates())).plot()#去重
# plt.title("每月消费人数")

user_grouped = df.groupby(by='user_id')[['order_products', 'order_amount']].sum()
# print(user_grouped.describe())
# print('用户数量:',len(user_grouped))
# df.plot(kind="scatter",x='order_products',y='order_amount')
# plt.show()

# plt.figure(figsize=(12,4)) 
# plt.subplot(121)

# df['order_amount'].plot(kind='hist',bins=50) 
# plt.xlabel('每个订单的消费金额') #bins:区间分数，影响柱子的宽度，值越大柱子越细。宽度=（列最大值-最小值）/bins
# #消费金额在100以内的订单占据了绝大多数

# plt.subplot(122)

# df.groupby(by='user_id')['order_products'].sum().plot(kind='hist',bins=50)
# plt.xlabel('每个uid购买的数量')
# #图二可知，每个用户购买数量非常小，集中在50以内
# plt.show()

# user_cumsum=df.groupby(by='user_id')['order_amount'].sum().sort_values().reset_index()
# user_cumsum['amount_cumsum']=user_cumsum['order_amount'].cumsum()
# user_cumsum.tail()

# amount_total=user_cumsum['amount_cumsum'].max()
# user_cumsum['prop']=user_cumsum.apply(lambda x:x['amount_cumsum']/amount_total,axis=1)
# print(user_cumsum.tail())
# user_cumsum['prop'].plot()
# plt.title('用户贡献')
# plt.show()
#第一次/最后一次购买时间
# plt.subplot(1,2,1)
# df.groupby(by='user_id')['order_date'].min().value_counts().plot()
# plt.subplot(1,2,2)
# df.groupby(by='user_id')['order_date'].max().value_counts().plot()
# plt.show()


#用户分层
rfm=df.pivot_table(index='user_id',
                   values=['order_products','order_amount','order_date'],
                   aggfunc={
                       'order_date':'max',#最后一次购买
                       'order_products':'sum',#购买产品的总数量
                       'order_amount':'sum'#消费总金额
                   })
rfm.head()
# print(rfm.head())
# print('='*32)
rfm['order_date'].max()
rfm['R']=-(rfm['order_date']-rfm['order_date'].max())/np.timedelta64(1,'D')
rfm.rename(columns={'order_products':'F','order_amount':'M'},inplace=True)
rfm.head() 

def rfm_func(x):
    level=x.apply(lambda x: '1'if x>=1 else '0')
    label=level['R']+level['F']+level['M']
    d={
        '111':'重要价值客户',
        '011':'重要保持客户',
        '101':'重要发展客户',
        '001':'重要挽留客户',
        '110':'一般价值客户',
        '010':'一般保持客户',
        '100':'一般发展客户',
        '000':'一般挽留客户'
    }
    result=d[label]
    return result

rfm['label']=rfm[['R','F','M']].apply(lambda x: x-x.mean()).apply(rfm_func,axis=1)
# print(rfm.head())

# for label,grouped in rfm.groupby(by='label'):
#     x=grouped['F']
#     y=grouped['R']
#     plt.scatter(x,y,label=label)
# plt.legend()
# plt.xlabel('F')
# plt.ylabel('R')
# plt.show()

pivoted_counts=df.pivot_table(
    index='user_id',
    columns='month',
    values='order_dt',
    aggfunc='count'
).fillna(0)
pivoted_counts.head()
df_purchase=pivoted_counts.applymap(lambda x: 1 if x>0 else 0)
# print(df_purchase.head())


def active_status(data):#data:每一行数据（18列）
    status=[]
    for i in range(18):
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
    return pd.Series(status,df_purchase.columns)

purchase_states=df_purchase.apply(active_status,axis=1)
# purchase_states.head()

purchase_states_ct = purchase_states.replace('unreg',np.NaN).apply(lambda x:pd.value_counts(x))#把unreg状态用nan替换
# purchase_states_ct.fillna(0).T.plot.area(figsize=(12,6)) #填充nan之后，进行行列变换
# plt.show()
#每月中回流用户占比情况（占所有用户的比例）
# plt.figure(figsize=(12,6))
# rate = purchase_states_ct.fillna(0).T.apply(lambda x:x/x.sum(),axis=1)
# rate.index = rate.index.astype(str)  # 关键转换步骤  
# plt.plot(rate['return'],label='return')
# plt.plot(rate['active'],label='active')
# plt.legend()

# plt.show()

#用户购买周期
data1=pd.DataFrame({
    'a':[0,1,2,3,4,5],
    'b':[5,4,3,2,1,0]
})
data1.shift(axis=0)
data1.shift(axis=1)
#计算购买周期
order_diff=df.groupby(by='user_id').apply(lambda x:x['order_date']-x['order_date'].shift())
order_diff.describe()
# print(order_diff.describe())
# (order_diff/np.timedelta64(1,'D')).hist(bins=20)



#计算方式：用户最后一次购买日期(max)-第一次购买的日期(min)。如果差值==0，说明用户仅仅购买了一次
user_life=df.groupby('user_id')['order_date'].agg(['min','max'])
# (user_life['max']==user_life['min']).value_counts().plot.pie(autopct='%1.1f%%')

# plt.legend(['仅消费一次','多次消费'])
# plt.show()
(user_life['max']-user_life['min']).describe()

# plt.figure(figsize=(12,6))
# plt.subplot(121)
# ((user_life['max']-user_life['min'])/np.timedelta64(1,'D')).hist(bins=15)
# plt.title('所有用户生命周期直方图')
# plt.xlabel('生命周期天数')
# plt.ylabel('用户人数')

# plt.subplot(122)
# u_1 = (user_life['max']-user_life['min']).reset_index()[0]/np.timedelta64(1,'D')
# u_1[u_1>0].hist(bins=15)
# plt.title('多次消费的用户生命周期直方图')
# plt.xlabel('生命周期天数')
# plt.ylabel('用户人数')
# plt.show()

#  # 转置使月份作为列  
# pivoted_counts.columns= pivoted_counts.columns.astype(str)  # 转换为字符串  
# purchase_r=pivoted_counts.applymap(lambda x:1 if x>1 else np.nan if x==0 else 0)
# purchase_r.head()
# (purchase_r.sum()/purchase_r.count()).plot(figsize=(12,6))

# #回购率分析
# def purchase_back(data):
#     status=[]
#     for i in range(17):
#         if data[i]==1:
#             if data[i+1]==1:
#                 status.append(1)
#             elif data[i+1]==0:
#                 status.append(0)
#         else:
#             status.append(np.nan)
#     status.append(np.nan)
#     return pd.Series(status,df_purchase.columns)

# purchase_b=df_purchase.apply(purchase_back,axis=1)

# print(purchase_b.head())

# plt.figure(figsize=(20,4))
# plt.subplot(211)
# (purchase_b.sum()/purchase_b.count()).plot(label='回购率')
# (purchase_r.sum()/purchase_r.count()).plot(label='复购率')
# plt.legend()
# plt.ylabel('百分比%')
# plt.title('用户回购率和复购率对比')

# plt.subplot(212)
# plt.plot(purchase_b.sum(),label='回购人数')
# plt.plot(purchase_r.count(),label='购物总人数')
# plt.xlabel('month')
# plt.ylabel('人数')
# plt.legend()
# plt.show()

# 核心解决方案：将Period索引转换为字符串  
pivoted_counts.columns = pivoted_counts.columns.astype(str)  # 转换为字符串  
  
# 创建purchase_r  
purchase_r = pivoted_counts.applymap(lambda x: 1 if x>1 else np.nan if x==0 else 0)  
  
# 创建模拟的df_purchase  
df_purchase = pivoted_counts.copy()  
  
# 修正purchase_back函数  
def purchase_back(data):  
    status = []  
    # 根据数据长度动态调整循环次数  
    for i in range(len(data)-1):  
        if data.iloc[i] == 1:  
            if data.iloc[i+1] == 1:  
                status.append(1)  
            elif data.iloc[i+1] == 0:  
                status.append(0)  
        else:  
            status.append(np.nan)  
    # 确保列表长度与原始数据一致  
    status += [np.nan] * (len(data) - len(status))  
    return pd.Series(status, index=data.index)  
  
# 应用purchase_back函数  
purchase_b = df_purchase.apply(purchase_back, axis=1)  
  
# 绘制图表  
plt.figure(figsize=(20,6))  
  
# 子图1：回购率和复购率对比  
plt.subplot(2,1,1)  
(purchase_b.sum()/purchase_b.count()).plot(label='回购率')  
(purchase_r.sum()/purchase_r.count()).plot(label='复购率')  
plt.legend()  
plt.ylabel('百分比%')  
plt.title('用户回购率和复购率对比')  
  
# 子图2：回购人数和购物总人数  
plt.subplot(2,1,2)  
plt.plot(pd.to_datetime(purchase_b.sum().index), purchase_b.sum(), label='回购人数')  
plt.plot(pd.to_datetime(purchase_r.count().index), purchase_r.count(), label='购物总人数')  
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  
plt.gcf().autofmt_xdate()  
plt.xlabel('月份')  
plt.ylabel('人数')  
plt.legend()  
  
# 保存并显示图表  
# plt.tight_layout()  

plt.show()