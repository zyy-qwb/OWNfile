import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import datetime
plt.rcParams['font.sans-serif'] = ['SimHei','DejaVu Sans']
INPUT = "Online Retail.xlsx"   # 改为你的路径
OUT = "rfm_output"
os.makedirs(OUT, exist_ok=True)



#只读取必要列以加速（如果 xlsx 很大，建议先转 csv）
usecols = ['InvoiceNo','InvoiceDate','Quantity','UnitPrice','CustomerID','Country','StockCode']
df = pd.read_excel(INPUT, engine='openpyxl', usecols=usecols, parse_dates=['InvoiceDate'])
#基本清洗
df.columns = df.columns.str.strip()
df = df.dropna(subset=['CustomerID'])#去除空ID
df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['IsReturn'] = df['InvoiceNo'].astype(str).str.startswith('C') | (df['Quantity'] < 0)#标记退货单和异常数量
# df['Date']=pd.to_datetime(df['InvoiceDate'],format='%Y%m%d')
df['month']=df['InvoiceDate'].dt.to_period('M')

#sales为去除退货订单数据，纯营业额
sales = df[~df['IsReturn']].copy()
sales = sales[(sales['Quantity'] > 0) & (sales['UnitPrice'] > 0)]

print(sales.info())
#1.计算退货率并绘图
return_rate=df['IsReturn'].mean()
non_return=1-return_rate
sizes=[return_rate,non_return]
labels=['退货','未退货']
plt.pie(sizes,labels=labels,autopct='%1.1f%%',startangle=90)
plt.title('退货率整体占比')
plt.axis('equal')
plt.legend()
plt.show()


#2.计算客单价
#计算每月客单价
average=sales.groupby('month').agg({
    'TotalPrice':'sum',
    'Quantity':'sum'
}).reset_index()
average['average']=average['TotalPrice']/average['Quantity']

#计算每个用户客单价
customer_average=sales.groupby('CustomerID').agg({
    'TotalPrice':'sum',
    'Quantity':'sum'
}).reset_index()
customer_average['customer_average']=customer_average['TotalPrice']/customer_average['Quantity']
print(average[['month','average']])
# customer=customer_average[['CustomerID','customer_average']]
customer_average.to_excel("客户单价表.xlsx",index=False)



#3.时间列表分析
#用户整体消费趋势分析（按月份）
plt.figure(figsize=(20,8))
#每月的产品购买数量
plt.subplot(2,2,1)
sales.groupby(by='month')['Quantity'].sum().plot()
plt.title('每月的产品购买数量')
#每月的消费金额
plt.subplot(2,2,2)
sales.groupby(by='month')['TotalPrice'].sum().plot()
plt.title('每月的消费金额')
plt.xlabel('月份')
plt.ylabel('销售额 (英镑)')
#每月的消费次数
plt.subplot(2,2,3)
sales.groupby(by='month')['InvoiceNo'].count().plot()
plt.title('每月的消费次数')
#每月的消费人数
plt.subplot(2,2,4)
sales.groupby(by='month')['CustomerID'].count().plot()
plt.title('每月的消费人数')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
plt.show()
# plt.tight_layout()  # 避免子图重叠  
# plt.savefig(os.path.join(OUT, 'monthly_trends.png'))  # 保存图片  
plt.close()  # 关闭画布释放内存


# 5.按国家聚合销售额
country_sales = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)

# 绘图
plt.figure(figsize=(10,6))
plt.bar(country_sales.index, country_sales.values, color='#66b3ff')
plt.title('销售额前十国家', fontsize=14)
plt.xlabel('国家', fontsize=12)
plt.ylabel('销售额（英镑）', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#6.用户消费
customer=sales.groupby('CustomerID').count()
print(customer.describe())
print('用户数量:',len(customer))

sales.plot(kind='scatter',x='Quantity',y='TotalPrice')
plt.figure(figsize=(12,4))
plt.subplot(121)
sales.groupby('InvoiceNo')['TotalPrice'].sum().plot(kind='hist',bins=50)
plt.xscale('log')
plt.xlabel('每个订单的消费金额')

plt.subplot(122)
sales.groupby('CustomerID')['Quantity'].sum().plot(kind='hist',bins=50)
plt.xlabel('每个客户购买的数量')
plt.xscale('log')
plt.show()

print(sales.groupby('InvoiceNo')['TotalPrice'].sum().max())
print(sales.groupby('CustomerID')['Quantity'].sum().max())

user_cumsum = df.groupby(by='CustomerID')['TotalPrice'].sum().sort_values().reset_index()
user_cumsum['amount_cumsum'] = user_cumsum['TotalPrice'].cumsum()
amount_total = user_cumsum['amount_cumsum'].max() #消费金额总值
user_cumsum['prop'] = user_cumsum.apply(lambda x:x['amount_cumsum']/amount_total,axis=1)  #前xx名用户的总贡献率
print(user_cumsum.tail())
user_cumsum['prop'].plot()
plt.title('用户贡献')
plt.show()

#用户分组，取最小值，即为首购时间，
plt.figure(figsize=(12,4))
plt.subplot(121)
sales.groupby(by='CustomerID')['InvoiceDate'].min().value_counts().plot()
plt.title('用户首购')
plt.subplot(122)
sales.groupby(by='CustomerID')['InvoiceDate'].min().value_counts().plot()
plt.title('用户最后一次购买')
plt.show()

#7.透视表，进行RFM用户分层
rfm = df.pivot_table(index='CustomerID',
                    values=['InvoiceNo','TotalPrice','InvoiceDate'],
                    aggfunc={
                        'InvoiceDate':'max',# 最后一次购买
                        'InvoiceNo':'nunique',# 订单数量
                        'TotalPrice':'sum'  #消费总金额
                        })
rfm['InvoiceDate'].max()
# 用每个用户的最后一次购买时间-日期列中的最大值，最后再转换成天数，小数保留一位
rfm['R'] = -(rfm['InvoiceDate']-rfm['InvoiceDate'].max())/np.timedelta64(1,'D')  #取相差的天数，保留一位小数
rfm.rename(columns={'InvoiceNo':'F','TotalPrice':'M'},inplace=True)

#构建 RFM 基础表（以 CustomerID 为索引）
rfm = df.pivot_table(
    index='CustomerID',
    values=['InvoiceNo','TotalPrice','InvoiceDate'],
    aggfunc={
        'InvoiceDate': 'max',   # 最近一次购买日期
        'InvoiceNo': 'nunique', # 购买次数（订单数）
        'TotalPrice': 'sum'     # 消费总额
    }
)

#计算 R 值（最近一次购买距今的天数）
max_date = rfm['InvoiceDate'].max()
rfm['R'] = (max_date - rfm['InvoiceDate']) / np.timedelta64(1, 'D')

#重命名 F、M 字段
rfm.rename(columns={'InvoiceNo': 'F', 'TotalPrice': 'M'}, inplace=True)


#把索引还原成普通列
rfm_value=rfm.reset_index()

# 保留关键字段
rfm_value=rfm_value[['CustomerID', 'R', 'F', 'M']]


# top10 = rfm.nlargest(10,'M')  # 先取 top10（如果你已经有了）
top10 =rfm_value.nlargest(10,'M').copy()  # 注意 copy()，以免 SettingWithCopyWarning
top10.to_excel('top10客户.xlsx',index=False)


# 在 top3 上添加 log 列
top10['F_log'] = np.log1p(top10['F'])
top10['M_log'] = np.log1p(top10['M'])

# 同时为 rfm 也创建 log 列用于绘制（如果还没有）
if 'F_log' not in rfm_value.columns:
    rfm_value['F_log'] = np.log1p(rfm_value['F'])
if 'M_log' not in rfm_value.columns:
    rfm_value['M_log'] = np.log1p(rfm_value['M'])

# 绘图（同上）
plt.figure(figsize=(8,6))
plt.scatter(rfm_value['F_log'], rfm_value['M_log'], alpha=0.5, label='其他客户')
plt.scatter(top10['F_log'], top10['M_log'], color='red', s=80, label='Top 3 客户')
for idx, row in top10.iterrows():
    plt.text(row['F_log'] + 0.02, row['M_log'], str(row['CustomerID']), color='red', fontsize=10)
plt.xlabel('log(1+F)')
plt.ylabel('log(1+M)')
plt.title('客户分布(log)—标出Top10客户')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

rfm_no_outlier = rfm_value[~rfm_value['CustomerID'].isin(top10['CustomerID'])]
plt.scatter(rfm_no_outlier ['F'], rfm_no_outlier ['M'], alpha=0.5)
plt.title('其他客户')
plt.show()

plt.scatter(top10['F'],top10['M'])
plt.title('top10客户')
plt.show()


#RFM计算方式：每一列数据减去数据所在列的平均值，有正有负，根据结果值与1做比较，如果>=1,设置为1，否则0
def rfm_func(x):  #x:分别代表每一列数据
    level = x.apply(lambda x:'1' if x>=1 else '0')
    level_2=x.apply(lambda x:'0' if x>=1 else '1')
    label = level_2['R'] + level['F'] + level['M']  #举例：100    001
    d = {
        '111':'重要价值客户',
        '011':'重要保持客户',
        '101':'重要发展客户',
        '001':'重要挽留客户',
        '110':'一般价值客户',
        '010':'一般保持客户',
        '100':'一般发展客户',
        '000':'一般挽留客户'
        
    }
    result = d[label]
    return result
# rfm['R']-rfm['R'].mean()
rfm['label'] = rfm[['R','F','M']].apply(lambda x:x-x.mean()).apply(rfm_func,axis =1)
print(rfm.head())


rfm_value['label'] = rfm_value[['R','F','M']].apply(lambda x:x-x.mean()).apply(rfm_func,axis =1)
rfm_grouped = rfm_value.groupby('label')[['R','F','M']].mean()

rfm_grouped.plot(kind='bar', figsize=(10,6))
plt.title('各客户分层平均 R/F/M 值')
plt.ylabel('平均值')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.5)
plt.show()



rfm_label_counts = rfm['label'].value_counts()

plt.figure(figsize=(12,4))
plt.pie(
    rfm_label_counts.values, 
    labels=None, 
    autopct=None, 
    startangle=90,
    colors=plt.cm.Paired.colors,
    # pctdistance=1.2,

)

  
# 计算总用户数和百分比  
total_users = rfm_value['CustomerID'].nunique() 
# print(total_users )
percentages = (rfm_label_counts / total_users)*100
plt.title('RFM 客户标签占比', fontsize=14)
# 创建自定义图例（合并标签和百分比）  
legend_data = []  
for i, label in enumerate(rfm_label_counts.index):  
    percent = percentages[i]  
    legend_data.append(f"{label} ({percent:.1f}%)")  
  
# 添加图例（右侧垂直排列）  
plt.legend(  
    legend_data,  
    title="客户标签",  
    # loc='center left',  
    # bbox_to_anchor=(1, 0.5), 
    fontsize=10,  
    frameon=True,  
    shadow=True  
)
plt.axis('equal')
plt.show()




#客户分层可视化
for label,grouped in rfm.groupby(by='label'):
#     print(label,grouped)
    x = grouped['F']  # 单个用户的订单数量
    y = grouped['R']  #最近一次购买时间与98年7月的相差天数
    plt.scatter(x,y,label=label)
plt.legend()  #显示图例
plt.title('F/R')
plt.xlabel('F')
plt.ylabel('R')
plt.show()


# 8.定义新老客户与活跃状态
cut_date = df['InvoiceDate'].max() - pd.Timedelta(days=90)
df_cust = df.groupby('CustomerID')['InvoiceDate'].agg(['min','max'])
df_cust['is_new'] = df_cust['min'] > cut_date
df_cust['is_active'] = df_cust['max'] > cut_date

matrix = pd.crosstab(df_cust['is_new'], df_cust['is_active'])
matrix.index = ['老客户','新客户']
matrix.columns = ['不活跃','活跃']

plt.figure(figsize=(6,5))
plt.imshow(matrix, cmap='Blues')
plt.xticks(range(2), matrix.columns)
plt.yticks(range(2), matrix.index)
for i in range(2):
    for j in range(2):
        plt.text(j, i, matrix.iloc[i,j], ha='center', va='center', color='black', fontsize=12)
plt.title('新/老 x 活跃/不活跃客户矩阵图')
plt.colorbar(label='客户数量')
plt.show()


# 按月统计客户是否复购
# 绘制留存率曲线
# 计算每个用户的首购月（CohortMonth）
first_purchase = df.groupby('CustomerID')['month'].min().reset_index()
first_purchase.columns = ['CustomerID', 'CohortMonth']

# 把 CohortMonth 合并回主表
df = df.merge(first_purchase, on='CustomerID')

# 计算 CohortIndex（用户在购买后第几个月）
# 计算月差： (year_diff * 12 + month_diff)
def month_diff(a, b):
    return (a.dt.year - b.dt.year) * 12 + (a.dt.month - b.dt.month)

df['CohortIndex'] = month_diff(df['month'], df['CohortMonth'])

# 构建 cohort 尺寸表：每个 CohortMonth 在第 k 月的活跃用户数（去重）
cohort_counts = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()

# 旋转为矩阵：行 = CohortMonth，列 = CohortIndex
cohort_pivot = cohort_counts.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')

# 计算留存率矩阵：每行除以第0列（首月用户数）
cohort_size = cohort_pivot.iloc[:,0]
retention = cohort_pivot.divide(cohort_size, axis=0).round(3)  # 留存率保留3位

# 绘制留存热力图（matplotlib）
plt.figure(figsize=(12,8))
plt.imshow(retention.fillna(0).values, aspect='auto', cmap='YlGnBu', interpolation='nearest')
plt.colorbar(label='Retention Rate')
plt.title('同期群留存率热力图')
plt.xlabel('同期群指数/月')
plt.ylabel('同期群月份')
# x ticks
plt.xticks(np.arange(retention.shape[1]), retention.columns)
# y ticks: format datetime nicely
ylabels = [d.strftime('%Y-%m') for d in retention.index]
plt.yticks(np.arange(retention.shape[0]), ylabels)
# 在每个单元格写入百分比
for i in range(retention.shape[0]):
    for j in range(retention.shape[1]):
        val = retention.iloc[i,j]
        if not np.isnan(val):
            plt.text(j, i, f"{val:.0%}", ha='center', va='center', color='black', fontsize=9)
plt.tight_layout()
plt.title('留存热力图')
plt.show()

# 绘制留存率曲线（选择若干 cohort）
plt.figure(figsize=(10,6))
# 选择例如最近12个 cohort 或全部前 12 条
selected = retention.iloc[:12]  # 前12个 cohort month（根据数据调整）
for idx, row in selected.iterrows():
    plt.plot(row.index, row.values, marker='o', label=idx.strftime('%Y-%m'))
# 也画平均留存率曲线
mean_ret = retention.mean(axis=0)
plt.plot(mean_ret.index, mean_ret.values, marker='x', linewidth=3, color='black', label='Average retention', linestyle='--')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title('留存率曲线（横截面）')
plt.xlabel('Months since first purchase')
plt.ylabel('留存率')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

purchase_intervals = df.sort_values(['CustomerID','InvoiceDate']).groupby('CustomerID')['InvoiceDate'].diff().dropna().dt.days

plt.figure(figsize=(8,5))
plt.hist(purchase_intervals, bins=30, color='#66b3ff', edgecolor='black')
plt.title('平均复购间隔直方图')
plt.xlabel('复购间隔（天）')
plt.ylabel('客户数')
plt.show()

product_sales = sales.groupby('StockCode')['TotalPrice'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
# 准备绘图的数值位置与标签（注意倒序使最高在上方）
labels = product_sales.index.tolist()[::-1]      # 列表，倒序用于 barh
values = product_sales.values.tolist()[::-1]     # 数值倒序

y_pos = np.arange(len(labels))   # 数值位置 [0..n-1]
plt.barh(y_pos, values, color='#ff9999', edgecolor='k')
plt.yticks(y_pos, labels)       # 将数值位置替换成商品名称
plt.xlabel('销售额（英镑）')
plt.title('热销商品 TOP10')
plt.tight_layout()

# 在条形右侧加数值标签（可选）
for i, v in enumerate(values):
    plt.text(v + max(values)*0.005, i, f'{v:,.0f}', va='center', fontsize=9)

plt.show()


plt.boxplot(np.log1p(df['UnitPrice']), vert=True, patch_artist=True, boxprops=dict(facecolor='#99ccff'))
plt.title('单价分布箱线图（对数尺度）')
plt.ylabel('对数单价（英镑）')
plt.grid(axis='y', alpha=0.5)
plt.show()


#新老，活跃，回流用户分析
pivoted_counts = sales.pivot_table(
                index='CustomerID',
                columns ='month',
                values = 'InvoiceDate',
                aggfunc = 'count'
).fillna(0)



# 由于浮点数不直观，并且需要转成是否消费过即可，用0、1表示
df_purchase = pivoted_counts.applymap(lambda x:1 if x>0 else 0)
# apply:作用与dataframe数据中的一行或者一列数据
# applymap:作用与dataframe数据中的每一个元素
# map:本身是一个series的函数，在df结构中无法使用map函数，map函数作用于series中每一个元素的
def active_status(data): #data：每一行数据（共18列）
    status = [] #存储用户12个月的状态（new|active|unactive|return|unreg）
    for i in range(13):
        #判断本月没有消费==0
        if data[i] ==0:
            if len(status)==0: #前几个月没有任何记录（也就是97年1月==0）
                status.append('unreg')  
            else:#之前的月份有记录（判断上一个月状态）
                if status[i-1] =='unreg':#一直没有消费过
                    status.append('unreg')
                else:#上个月的状态可能是：new|active|unative|reuturn
                    status.append('unactive')
        else:#本月有消费==1
            if len(status)==0:
                status.append('new') #第一次消费
            else:#之前的月份有记录（判断上一个月状态）
                if status[i-1]=='unactive':
                    status.append('return') #前几个月不活跃，现在又回来消费了，回流用户
                elif  status[i-1]=='unreg':
                    status.append('new') #第一次消费
                else:#new|active
                    status.append('active') #活跃用户
            
    return pd.Series(status,df_purchase.columns) #值：status,列名：12个月份

purchase_states = df_purchase.apply(active_status,axis=1) #得到用户分层结果
#把unreg状态用nan替换
purchase_states_ct = purchase_states.replace('unreg',np.NaN).apply(lambda x:pd.value_counts(x))
purchase_states_ct.head()
#数据可视化，面积图
purchase_states_ct.fillna(0).T.plot.area(figsize=(12,6))  #填充nan之后，进行行列变换
#每月中回流用户占比情况（占所有用户的比例）
plt.figure(figsize=(12,6))

rate = purchase_states_ct.fillna(0).T.apply(lambda x:x/x.sum(),axis=1)
rate.index = rate.index.astype(str) 
plt.plot(rate['return'],label='return')
plt.plot(rate['active'],label='active')
plt.title('return/active折线图')
plt.legend()
plt.show()


#InvoiceNo	StockCode	Description	Quantity	InvoiceDate	UnitPrice	CustomerID	Country

#用户的购买周期
order_diff = sales.groupby(by='CustomerID').apply(lambda x:x['InvoiceDate']-x['InvoiceDate'].shift()) #当前订单日期-上一次订单日期
(order_diff/np.timedelta64(1,'D')).hist(bins = 20) #影响柱子的宽度，  每个柱子的宽度=（最大值-最小值）/bins
plt.title('用户购买周期')
plt.show()
#用户生命周期
#计算方式：用户最后一次购买日期(max)-第一次购买的日期(min)。如果差值==0，说明用户仅仅购买了一次
user_life = sales.groupby('CustomerID')['InvoiceDate'].agg(['min','max'])
(user_life['max']==user_life['min']).value_counts().plot.pie(autopct='%1.1f%%') #格式化成1为小数
plt.legend(title=['仅消费一次','多次消费'],loc='upper left')
plt.axis('equal')
plt.title('消费次数分布')
plt.show()



(user_life['max']-user_life['min']).describe()  #生命周期分析
plt.figure(figsize=(12,6))
plt.subplot(121)
((user_life['max']-user_life['min'])/np.timedelta64(1,'D')).hist(bins=15)
plt.title('所有用户生命周期直方图')
plt.xlabel('生命周期天数')
plt.ylabel('用户人数')

plt.subplot(122)
u_1 = (user_life['max']-user_life['min']).reset_index()[0]/np.timedelta64(1,'D')
u_1[u_1>0].hist(bins=15)
plt.title('多次消费的用户生命周期直方图')
plt.xlabel('生命周期天数')
plt.ylabel('用户人数')
plt.show()

#复购率和回购率

#复购率计算方式：在自然月内，购买多次的用户在总消费人数中的占比（若客户在同一天消费了多次，也称之复购用户）
#消费者有三种：消费记录>=2次的；消费中人数；本月无消费用户；
#复购用户:1    非复购的消费用户：0   自然月没有消费记录的用户：NAN(不参与count计数)
purchase_r = pivoted_counts.applymap(lambda x: 1 if x>1 else np.NaN  if x==0 else 0)
purchase_r.head()
#purchase_r.sum() :求出复购用户
#purchase_r.count():求出所有参与购物的用户（NAN不参与计数）
(purchase_r.sum()/purchase_r.count()).plot(figsize=(12,6))

# 核心解决方案：将Period索引转换为字符串  
pivoted_counts.columns = pivoted_counts.columns.astype(str)  # 转换为字符串  
# 创建purchase_r  
purchase_r = pivoted_counts.applymap(lambda x: 1 if x>1 else np.nan if x==0 else 0)  
  
# 创建模拟的df_purchase  
df_purchase = pivoted_counts.copy() 

#回购率计算方式：在一个时间窗口内进行了消费，在下一个窗口内又进行了消费
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

# purchase_b.head()
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
plt.title('回购人数和购物总人数')
  
# 保存并显示图表  
# plt.tight_layout()  

plt.show()

