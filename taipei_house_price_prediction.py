#!/usr/bin/env python
# coding: utf-8

# In[1]:


#导入类库和加载数据集
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_names = ["transaction date",
               "house age",
               "distance to the nearest MRT station",
               "number of convenience stores",
               "latitude",
               "longitude",
               "house price of unit area"]
#读取数据
data = pd.read_csv(r'C:\Users\孙露云\Desktop\Real_estate_data.csv',names=train_names,header=1)
data.head()


# In[3]:


#拆分特征
transaction_year,transaction_month=[],[]
for [date] in data[['transaction date']].values:
    year,month=int(date),int(12*(date%1))+1
    transaction_year.append(year)
    transaction_month.append(month)
del data['transaction date']
data['transaction_year']=pd.DataFrame({'transaction_year':transaction_year})
data['transaction_month']=pd.DataFrame({'transaction_month':transaction_month})
data.head()


# In[4]:


# 检查数据集
data.info()

# 检查数据中是否有缺失值
#Flase:对应特征的特征值中无缺失值
#True：有缺失值
print(data.isnull().any())
#查看缺失值记录
#data_null = pd.isnull(data)
#data_null = data[data_null==True]
#print(data_null)
#缺失值处理:删除包含缺失值的行
data.dropna(inplace=True)

#检查是否包含无穷数据
#False:包含
#True:不包含
print(np.isfinite(data).all())

# 再次检查数据集
data.info()


# In[5]:


#数据描述性统计
print(data['house age'].describe())
print(data['distance to the nearest MRT station'].describe())
print(data['number of convenience stores'].describe())
print(data['latitude'].describe())
print(data['longitude'].describe())
print(data['transaction_year'].describe())
print(data['transaction_month'].describe())
print(data['house price of unit area'].describe())


# In[6]:


#观察数据分布
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def view_data(df_series,title,xlabel):
    #fig = plt.figure()
    #sns.kdeplot(df_series,shade=True)
    plt.figure(figsize = (10,5))
    plt.hist(df_series, bins = 30,color = 'steelblue',rwidth=0.2)
    #text = '峰度: ' + str(np.round(df_series.skew(),3)) +'\n偏度: ' +str(np.round(df_series.kurt(),3))
    #plt.title(text)
    plt.ylabel('频数')
    plt.xlabel(xlabel)
    plt.title(title)
view_data(data['house price of unit area'],'house price of unit area','单位面积房价（单位：万台币/每坪）')
view_data(data['house age'],'house age','房屋年龄（单位：年）')
view_data(data['distance to the nearest MRT station'],'distance to the nearest MRT station','到最近的捷运站的距离(单位:米)')
view_data(data['number of convenience stores'],'number of convenience stores','徒步生活圈的便利店数量(单位：个)')
view_data(data['transaction_year'],'transaction_year','交易年份（单位:年）')
view_data(data['transaction_month'],'transaction_month','交易年份（单位:月）')
view_data(data['latitude'],'latitude','纬度(单位:度)')
view_data(data['longitude'],'longitude','经度(单位:度)')


# In[7]:


#数据概览
import pandas_profiling
col =[ 'transaction_year','transaction_month','house age', 'distance to the nearest MRT station', 'number of convenience stores',
       'latitude',  'longitude', 'house price of unit area']
sns.pairplot(data[col])
plt.show()


# In[8]:


#自变量与因变量的相关性分析
plt.figure(figsize = (20,10))
internal_chars = ['house price of unit area','house age','distance to the nearest MRT station','number of convenience stores','latitude','longitude','transaction_year','transaction_month',]
corrmat = data[internal_chars].corr()  # 计算相关系数
sns.heatmap(corrmat, square=False, linewidths=.5, annot=True) #热力图
#打印出相关性的排名
print(corrmat["house price of unit area"].sort_values(ascending=False))


# In[9]:


#进一步分析相关性
sns.jointplot('house age','house price of unit area',data,kind='reg')
plt.show()
sns.jointplot('distance to the nearest MRT station','house price of unit area',data,kind='reg')
plt.show()
sns.jointplot('transaction_year','house price of unit area',data,kind='reg')
plt.show()
sns.jointplot('transaction_month','house price of unit area',data,kind='reg')
plt.show()
sns.jointplot('number of convenience stores','house price of unit area',data,kind='reg')
plt.show()
sns.jointplot('latitude','house price of unit area',data,kind='reg')
plt.show()
sns.jointplot('longitude','house price of unit area',data,kind='reg')
plt.show()


# In[10]:


#主成分分析
from sklearn.decomposition import PCA
model = PCA(n_components=4)
model.fit_transform(data)
exp_var = np.round(model.explained_variance_ratio_,decimals=5)
print('各个主成分解释的方差百分比',exp_var)


# In[11]:


#特征缩放（归一化）
data = data.astype('float')
x = data.drop('house price of unit area',axis=1)
y = data['house price of unit area']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
newX= scaler.fit_transform(x)
newX = pd.DataFrame(newX, columns=x.columns)
newX.head()


# In[12]:


#将数据集分成训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(newX, y, test_size=0.2, random_state=21)


# In[34]:


#模型建立
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor #随机森林 
model= RandomForestRegressor(n_estimators=155,max_features=None)
model.fit(X_train, y_train)
predicted_RF= model.predict(X_test)
mse = metrics.mean_squared_error(y_test,predicted_RF)
print(y_test)
print(predicted_RF)
print('RF mse: ',mse/10000)
#绘制随机森林预测房价的散点图
import matplotlib as mpl
x = range(len(predicted_RF))
plt.scatter(x,predicted_RF,label='predicted_RF')
plt.xlim(0,90)
plt.ylim(0,80)
#plt.show()
x = range(len(y_test))
plt.scatter(x,y_test,label='y_test')
#plt.xlim(0,100)
#plt.ylim(0,80)
plt.legend(loc=0)
plt.xlabel("样本序号")
plt.ylabel("单位面积的房价（单位：万台币/坪）")
plt.show()
   
from sklearn.linear_model import LinearRegression #线性回归           
LR = LinearRegression()
LR.fit(X_train, y_train)
predicted_LR= LR.predict(X_test)
mse = metrics.mean_squared_error(y_test,predicted_LR)
print(predicted_LR)
print('LR mse: ',mse/10000)
#绘制线性回归预测房价的散点图
x = range(len(predicted_LR))
plt.scatter(x,predicted_LR,label='predicted_LR')
plt.xlim(0,90)
plt.ylim(0,80)
#plt.show()

plt.scatter(x,y_test,label='y_test')
#plt.xlim(0,100)
#plt.ylim(0,80)
plt.legend(loc=0)
plt.xlabel("样本序号")
plt.ylabel("单位面积的房价（单位：万台币/坪）")
plt.show()


import xgboost as xgb#xgboost
regr = xgb.XGBRegressor(n_estimators=11)
regr.fit(X_train,y_train)
predicted_xgboost = regr.predict(X_test)
mse = metrics.mean_squared_error(y_test,predicted_xgboost)
print('XGboost MSE',mse/10000)

x = range(len(predicted_xgboost))
plt.scatter(x,predicted_xgboost,label='predicted_xgboost')
plt.xlim(0,90)
plt.ylim(0,80)
#plt.show()

plt.scatter(x,y_test,label='y_test')
#plt.xlim(0,100)
#plt.ylim(0,80)
plt.legend(loc=0)
plt.xlabel("样本序号")
plt.ylabel("单位面积的房价（单位：万台币/坪）")
plt.show()

