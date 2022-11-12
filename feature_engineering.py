#!/usr/bin/env python
# coding: utf-8

# # Retail Transaction and Promotion Response Feature Engineering
# ##### By: Portia Masibi

# #### Using Retail Transaction Data from Kaggle to build  a wide range of features  that will be used as inputs to predict  the clients response to a promotion campaign 

# ![image-3.png](attachment:image-3.png)

# In[1]:


#First to import relevant libraries and modules that will be used 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Importing data 
transaction_data = pd.read_csv('Retail_Data_Transactions.csv')
print( 'There are',len(transaction_data), 'rows')
transaction_data


# In[3]:


print( 'Checking for missing values',transaction_data.isnull().sum())
# Checking the data types we are dealing with
transaction_data.info()


# In[4]:


# Getting statistical data from the transaction amount
transaction_data.describe()


# In[5]:


# First changing the date column to standard time, making it easier to deal with time features
transaction_data["txn_date"] = pd.to_datetime(transaction_data["trans_date"])
transaction_data = transaction_data.drop(columns ="trans_date")
transaction_data


# In[6]:


# Min and Max of txn_date
min_date = min(transaction_data["txn_date"])
max_date = max(transaction_data["txn_date"])
print('The min of the transaction dates is:', min_date , '\n')
print('The max of the transaction dates is:', max_date)


# In[7]:


# Column for the last day of the month 
transaction_data["ME_DT"] = transaction_data["txn_date"] + pd.offsets.MonthEnd(0)
transaction_data


# In[8]:


# Column with year 
transaction_data["YEAR"] = pd.DatetimeIndex(transaction_data["txn_date"]).year
transaction_data


# ### 1. Features that Capture Annual Spending

# #### Capturing the clients annual spending by using the rationale that the clients spend is not very frequent to capture in a monthly aggregation 

# In[9]:


# Creating an annual aggregrations dataframe with its information

column_names = (['ann_txn_amt_sum','ann_txn_amt_ave','ann_txn_amt_std','ann_txn_amt_var','ann_txn_amt_sem','ann_txn_amt_max','ann_txn_amt_min','ann_txn_cnt'])
clnt_annual_aggregations = transaction_data.groupby(['customer_id', 'YEAR']).agg(['sum','mean','std','var','sem','max','min','count'])
clnt_annual_aggregations.columns = column_names
clnt_annual_aggregations


# In[10]:


# Plotting histogram of the sum and count 

plt.hist(clnt_annual_aggregations['ann_txn_amt_sum'])
plt.title('Annual Transaction Amount Sum')
plt.show()

plt.hist(clnt_annual_aggregations['ann_txn_cnt'])
plt.title('Annual Transcation Count')
plt.show()


# In[11]:


# Pivoting the table 
clnt_annual_aggregations_pivot = clnt_annual_aggregations.unstack(1)
print('\n')
print('There are 40 columns as for every year, 2011-2015, the annual  sum, mean, std, var, sem, max, min, count is calculated, so 5 years and 8 statistical/descriptive calculation which give 40 columns ')
clnt_annual_aggregations_pivot 


# In[12]:


# Replacing NaN values 
print('\n')
print('Replacing NaN with zero as the frequency of customers buying may not be yearly, so the sum can be zero, which makes other variables 0 too \n')
print('For Example this can be seen in the year 2015,customer CS116, the sum is NaN,meaning they did not buy that year so mean, std, var, sem, max, min, count are also NaN. This can simply be replaced by 0.')
clnt_annual_aggregations_pivot = clnt_annual_aggregations_pivot.fillna(value = 0)
clnt_annual_aggregations_pivot


# In[13]:


# Number of levels and columns 

print('\n')
print('There are 2 levels.For every column in the 1st level, there are 5 columns in the 2nd level, so as we have 8 columns in the first level and 5 for each column in the second level this gives us 40 columns \n')
print('Number of levels:',clnt_annual_aggregations_pivot . columns . nlevels)
clnt_annual_aggregations_pivot . columns


# In[14]:


#Saving clnt_annual_aggregations_pivot as an.xlsx file
level_0 = clnt_annual_aggregations_pivot . columns . get_level_values ( 0 ) .astype ( str)
level_1 = clnt_annual_aggregations_pivot . columns . get_level_values ( 1 ) .astype ( str)
clnt_annual_aggregations_pivot . columns = level_0 + '_' + level_1
clnt_annual_aggregations_pivot


# In[15]:


# Then save it, now we have annual features
clnt_annual_aggregations_pivot.to_excel('annual_features.xlsx')


# ### 2. Features that Capture Monthly Spending 

# #### In this section we  compare Montlhy and Annual Sum and Count of transactions 

# In[16]:


# Dataframe that captures monthly. sum and count of transactions per client
column_names = (['mth_txn_amt_sum','mth_txn_cnt'])
clnt_monthly_aggregations = transaction_data.groupby(['customer_id','ME_DT']).agg({'tran_amount':['sum','count']})
clnt_monthly_aggregations.columns = column_names
clnt_monthly_aggregations.head(15) 


# In[17]:


# Histogram of both columns
plt.hist(clnt_monthly_aggregations['mth_txn_amt_sum'])
plt.title('Montlhy Transaction Amount Sum')
plt.show()

plt.hist(clnt_monthly_aggregations['mth_txn_cnt'])
plt.title('Montlhy Transcation Count')
plt.show()


# In[18]:


# Comparing yearly and monthly 

fig, axs = plt.subplots(2, 2, figsize=(15,15) )
axs[0, 0].hist(clnt_monthly_aggregations['mth_txn_amt_sum'])
axs[0, 0].set_title('Montlhy Transaction Amount Sum')

axs[0, 1].hist(clnt_monthly_aggregations['mth_txn_cnt'])
axs[0, 1].set_title('Montlhy Transcation Count')

axs[1, 0].hist(clnt_annual_aggregations['ann_txn_amt_sum'])
axs[1, 0].set_title('Annual Transaction Amount Sum')

axs[1, 1].hist(clnt_annual_aggregations['ann_txn_cnt'])
axs[1, 1].set_title('Annual Transcation Count')


print('\nThe count for some months is zero indicating that the customers do not  buy on a monthly basis')
print('The maximum monthly sum is ~ 50,000 while maximum monthly count is ~ 85,000')
print('The maximum annual sum is ~ 6,500 while maximum annual count is ~ 8,800 ')
print('The monthly values are larger than yearly values as they capture more data in a short timespan')
print('\nMost clients in this dataset shop a few times a year. For example, the client with ’customer id’ CS1112 shown above made purchases in 15 out of 47 months of data in the txn table. The information in this dataset is ”irregular”; some clients may have an entry for a month, whileothers do not have an entry (e.g. when they don’t shop for this particular month)')


# ### Create the monthly rolling window features

# #### This is to convert the irregular transaction data into the typical time series data; data captured at equal intervals. Feature engineering of time series data gives you the potential to build very powerful predictive models.

# ### First we create the base table for the rolling window features

# #### In order to create the rolling window features we need to create a base table with all possible combinations of ’customer id’ and ’ME DT’. For example, customer CS1112 should have 47 entries, one for each month, in which 15 will have the value of transaction amount and the rest 32 will have zero value for transaction amount. This will essentially help to convert the ”irregular” clnt monthly aggregations table into a ”regular” one.

# In[19]:


# Number of unique clients and unique month-end-dates
clnt_no = transaction_data["customer_id"]
me_dt = transaction_data['ME_DT']

print('Number of unique clients: ',clnt_no.nunique())
print('Number of unique month-end-dates: ',me_dt.nunique())


# In[20]:


# Using  itertools.product to generate all the possible combinations of ’customer id’ and’ME DT’. 
# Itertools is a Python module that iterates over data in a computationally efficient way

#from itertools import product
#base_table = product(clnt_no,me_dt)
#type(base_table)


# In[21]:


# Converting the itertools product object into a pandas object 
#base_table_pd = pd.DataFrame.from_records(base_table, columns = ['CLNT_NO','ME_DT'])


# In[22]:


# If the intertools take longer 

cln = transaction_data['customer_id'].unique().tolist()
cln.sort()
len(cln)

dt = transaction_data['ME_DT'].unique().tolist()
len(dt)

# creating an array that is 323,783 (47*6889) by 2 ( client,date)

bpt = []
for i in range(0,2000):
    for j in range(0,len(dt)):
        bpt.append([cln[i],dt[j]])
        
for i in range(2000,4000):
    for j in range(0,len(dt)):
        bpt.append([cln[i],dt[j]])
        
for i in range(2000,4000):
    for j in range(0,len(dt)):
        bpt.append([cln[i],dt[j]])


# In[25]:


base_table_pd = pd.DataFrame(bpt, columns = ['CLNT_NO','ME_DT'])
# converting the 'ME_DT' column to datetime format
base_table_pd['ME_DT']= pd.to_datetime(base_table_pd['ME_DT'])

base_table_pd


# In[26]:


# Validating that the table created is correct
# First checking the number of unique clients and number of unique month end dates 

# Confirming base_table_pd details 

print('Rows of the Base Table are:', len(base_table_pd))
print('Unique clients:', base_table_pd['CLNT_NO'].nunique())
print('Unique month ends:',base_table_pd['ME_DT'].nunique())


# In[27]:


# Then checking filtering a client to see if the min and max dates fall between 2011-05-16 00:00:00  and 2015-03-16 00:00:00

contain_values= base_table_pd[base_table_pd['CLNT_NO'].str.contains('CS1112')]   
CS1112 = pd.DataFrame(contain_values)

min_date_CS1112 = min(CS1112["ME_DT"])
max_date_CS1112 = max(CS1112["ME_DT"])

print('The min of the transaction dates for customer CS1112 is:', min_date_CS1112)
print('The max of the transaction dates for customer CS1112 is:', max_date_CS1112)
print('So the min and max month dates fall within the range \n')

print('Client CS1112 has',len(CS1112),'rows')
CS1112


# #### Now that we have the base table we can create the monthly rolling window features

# In[28]:


# Left-joining the base table pd with the clnt monthly aggregations table from section on 
# [CLNT NO, ME DT] to create the table base clnt mth

clnt_monthly_aggregations1 = clnt_monthly_aggregations.reset_index(level='ME_DT')
clnt_monthly_aggregations2 = clnt_monthly_aggregations1.reset_index(level='customer_id')
clnt_monthly_aggregations2  = clnt_monthly_aggregations2.rename(columns={'customer_id': 'CLNT_NO'})
clnt_monthly_aggregations2 


# In[29]:


# Joining base_table_pd with clnt_monthly_aggregations
base_clnt_mth = pd.merge(base_table_pd,clnt_monthly_aggregations2,on = ['CLNT_NO','ME_DT'],how='left')
base_clnt_mth


# In[30]:


print('Some rows on the merged dataframe have NaN as for those dates/months there are no transactions so the NaN will be replaced by 0 in both the sum and count values \n')
print('The dataframe has  rows as expected as it was merged to the base_table_pd with 323,783 rows \n')
print('The base_clnt_mth has 323,783 rows and the clnt_monthly_aggregation has 103,234, the difference is beacuse for the clnt_monthly_aggregation it only captures months of transactions while the base_clnt_month captures all months, with or without transactions')
base_clnt_mth = base_clnt_mth.fillna(0)
base_clnt_mth


# In[31]:


# Sorting client names and dates in ascending order, necessary for creating the order for rolling windows
base_clnt_mth = base_clnt_mth.sort_values(by = ['CLNT_NO','ME_DT'])
base_clnt_mth.head(50)


# In[32]:


# Using the rolling window to calculate statistical properties

# window_size = 3
indv_client = base_clnt_mth.groupby('CLNT_NO')
rolling_features_3M = indv_client.rolling(window = 3).agg(['sum','mean','max'])
rolling_features_3M


# In[33]:


#window_size = 6
rolling_features_6M = indv_client.rolling(window = 6).agg(['sum','mean','max'])
rolling_features_6M.head(10)


# In[34]:


#window_size = 12
rolling_features_12M = indv_client.rolling(window = 12).agg(['sum','mean','max'])
rolling_features_12M.head(15)


# 
# ##### We get NaN values  because when using rolling window, where the window size is n, the rolling looks for n-1 rows of data to aggregate, when the condition is not met, it will return NaN for the window. 
# 

# In[35]:


# Renaming columns and changing it to 1 level rom the 2 multi-index dataframe, makes it easier to save in excel

columns3M = ['amt_sum_3M','amt_mean_3M','amt_max_3M','txn_cnt_sum_3M','txn_cnt_mean_3M','txn_cnt_max_3M']
rolling_features_3M.columns = columns3M

columns6M = ['amt_sum_6M','amt_mean_6M','amt_max_6M','txn_cnt_sum_6M','txn_cnt_mean_6M','txn_cnt_max_6M']
rolling_features_6M.columns = columns6M

columns12M = ['amt_sum_12M','amt_mean_12M','amt_max12M','txn_cnt_sum_12M','txn_cnt_mean_12M','txn_cnt_max_12M']
rolling_features_12M.columns = columns12M
rolling_features_12M


# In[36]:


# Merging tables  with the base_clnt_mth
# first dropping index level:0 
rolling_features_3M = rolling_features_3M.droplevel(0)
rolling_features_6M = rolling_features_6M.droplevel(0)
rolling_features_12M = rolling_features_12M.droplevel(0)


# In[37]:


all_rolling_features = pd.merge(base_clnt_mth,rolling_features_3M,left_index=True, right_index=True)
all_rolling_features = pd.merge(all_rolling_features,rolling_features_6M,left_index=True, right_index=True)
all_rolling_features = pd.merge(all_rolling_features,rolling_features_12M,left_index=True, right_index=True)
all_rolling_features


# In[38]:


# Saving it as an xlsx file 
all_rolling_features.to_excel('mth_rolling_features.xlsx')


# ## 3. Date Related Features : Date of the Week 
# #### Date-related features that capture information about the day of the week the transactions were performed

# In[39]:


# The DatetimeIndex object allows extraction of many components of a DateTime object
# Here, we want to extract the day of the week from column ’txn date’ of the txn table (with Monday=0, Sunday=6)

day_of_the_week = transaction_data['txn_date'].dt.dayofweek
transaction_data['day_of_the_week'] = day_of_the_week 

day_name = transaction_data['txn_date'].dt.day_name()
transaction_data['day_name'] = day_name

transaction_data


# In[42]:


# Bar plot of count of transactions per day of the week 

x = transaction_data.groupby('day_name').count()
fig = plt.figure(figsize=(10,5))
plt.bar(x.index,x['day_of_the_week'])

print('\nFrom the figure below, thr transcations per day are approximately evenly distributed')


# In[43]:


# 1.6.3 Capturing the count of transactions per client,year and day of the week

clnt_daily_aggregations = transaction_data.groupby(['customer_id','YEAR','day_name']).count()
clnt_daily_aggregations = clnt_daily_aggregations.unstack(1)
clnt_daily_aggregations = clnt_daily_aggregations.unstack(1)
clnt_daily_aggregations = clnt_daily_aggregations['tran_amount']
clnt_daily_aggregations = clnt_daily_aggregations.fillna(0)
clnt_daily_aggregations


# In[44]:


# Dropping a level from a 3 level multi-level dataframe and remaning columns 
clnt_daily_aggregations = clnt_daily_aggregations.droplevel(0,axis = 1)
clnt_daily_aggregations


# In[45]:


# Adding column names 

columns_daily =['cnt_2011_Friday','cnt_2011_Monday','cnt_2011_Saturday','cnt_2011_Sunday','cnt_2011_Thursday','cnt_2011_Tuesday','cnt_2011_Wednesday',
                'cnt_2012_Friday','cnt_2012_Monday','cnt_2012_Saturday','cnt_2012_Sunday','cnt_2012_Thursday','cnt_2012_Tuesday','cnt_2012_Wednesday',
                'cnt_2013_Friday','cnt_2013_Monday','cnt_2013_Saturday','cnt_2013_Sunday','cnt_2013_Thursday','cnt_2013_Tuesday','cnt_2013_Wednesday',
                'cnt_2014_Friday','cnt_2014_Monday','cnt_2014_Saturday','cnt_2014_Sunday','cnt_2014_Thursday','cnt_2014_Tuesday','cnt_2014_Wednesday',
                'cnt_2015_Friday','cnt_2015_Monday','cnt_2015_Saturday','cnt_2015_Sunday','cnt_2015_Thursday','cnt_2015_Tuesday','cnt_2015_Wednesday']


clnt_daily_aggregations.columns = columns_daily


# In[48]:


#Confirming that output has the same number of rows as the annual features
print('clnt_annual_aggregations_pivot length:',len(clnt_annual_aggregations_pivot))
print('clnt_daily_aggregations length:', len(clnt_daily_aggregations))
# Therefore same number of rows 


# In[49]:


# saving as excel file with 35 features/ columns 
clnt_daily_aggregations.to_excel('annual_day_of_week_counts_pivot.xlsx')


# In[50]:


# Creating Features that capture the count of transactions per client,month-end-date and day of the week 

clnt_daily_month_aggregations = transaction_data.groupby(['customer_id','ME_DT','day_name']).count()
clnt_daily_month_aggregations = clnt_daily_month_aggregations.unstack(2)
clnt_daily_month_aggregations = clnt_daily_month_aggregations['tran_amount']
clnt_daily_month_aggregations = clnt_daily_month_aggregations.fillna(0)
clnt_daily_month_aggregations

columns_daily_month = ['cnt_Friday','cnt_Monday','cnt_Saturday','cnt_Sunday','cnt_Thursday','cnt_Tuesday','cnt_Wednesday']

clnt_daily_month_aggregations.columns = columns_daily_month
clnt_daily_month_aggregations


# In[51]:


# Joining with base_table 
clnt_daily_month_aggregations1 = clnt_daily_month_aggregations.reset_index(level='ME_DT')
clnt_daily_month_aggregations2 = clnt_daily_month_aggregations1.reset_index(level='customer_id')
clnt_daily_month_aggregations2  = clnt_daily_month_aggregations2.rename(columns={'customer_id': 'CLNT_NO'})
clnt_daily_month_aggregations2


# In[54]:


monthly_day_counts = pd.merge(base_table_pd,clnt_daily_month_aggregations2,on = ['CLNT_NO','ME_DT'],how='left')
monthly_day_counts = monthly_day_counts .fillna(0)
monthly_day_counts.to_excel('mth_day_counts.xlsx')


# ### 4. Date-related Features: Days Since Last Transaction

# #### In this date-related features set,we capture the frequency of the transactions in terms of the days since the last transaction. This set of features applies only to the monthly features.

# In[55]:



# Capturing last monthly purchase 
# The starting point is the txn table. Recall that most clients have a single purchase per month, but some clients 
# have multiple purchases in a month. Since we want to calculate the ”days since last transaction”, we want to capture
# the last transaction in a month for every client

last_monthly_purchase = transaction_data.groupby(['customer_id','ME_DT']).max()
last_monthly_purchase = last_monthly_purchase['txn_date']
last_monthly_purchase


# In[59]:


#Joining base table pd with last monthly purchase
last_monthly_purchase1 = last_monthly_purchase.reset_index(level='ME_DT')
last_monthly_purchase2 = last_monthly_purchase1.reset_index(level='customer_id')
last_monthly_purchase2  = last_monthly_purchase2.rename(columns={'customer_id': 'CLNT_NO'})
last_monthly_purchase2 = last_monthly_purchase2.rename(columns={'txn_date': 'last_monthly_purchase'})


base_table_pd_sorted = base_table_pd.sort_values(by = ['CLNT_NO','ME_DT'])


last_monthly_purchase_base = pd.merge(base_table_pd_sorted,last_monthly_purchase2,on = ['CLNT_NO','ME_DT'],how='left')
last_monthly_purchase_base


# In[64]:


# Filling NaT 
last_monthly_purchase_base = last_monthly_purchase_base.groupby('CLNT_NO').apply(lambda x: x.ffill())

last_monthly_purchase_base.loc[92:98]


# In[65]:



# Computing days since last transaction
# Subtracting the two date columns to calculate the column ’days since last txn’ 
days_since_last_txn = last_monthly_purchase_base['ME_DT'] - last_monthly_purchase_base['last_monthly_purchase']

days_since_last_txn = days_since_last_txn.dt.days

days_since_last_txn 

last_monthly_purchase_base['days_since_last_txn'] = days_since_last_txn 

last_monthly_purchase_base.head(60)


# In[66]:


# Histogran for days_since_last_txn
plt.hist(last_monthly_purchase_base['days_since_last_txn'], bins = 10)
plt.title('Days Since Last Transactions')
plt.show()


# In[67]:


print('The days with the common density is 100 so this is used to fill in number of days since last transaction values that give NaN')
last_monthly_purchase_base['days_since_last_txn'] = last_monthly_purchase_base['days_since_last_txn'].fillna(100)
last_monthly_purchase_base


# In[68]:


# Saving to excel
days_since_last_txn_excel = last_monthly_purchase_base.drop(['last_monthly_purchase'], axis = 1)
days_since_last_txn_excel.to_excel('days_since_last_txn.xlsx')


# ### RECAP : We created new features to use in our models, which are 
# #### 1. Features that capture Annual Spending
# #### 2. Features that Capture Monthly Spendingrolling features
# #### 3. Date Related Features 
# #### 4. Date-related Features

# In[ ]:




