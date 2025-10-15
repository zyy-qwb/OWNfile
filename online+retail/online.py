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

# 5.按国家聚合销售额
country_sales = sales.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
country_sales_df=country_sales.reset_index(name='销售额')
print(country_sales_df)
country_sales_df.to_excel('top10_country.xlsx',index=False)
