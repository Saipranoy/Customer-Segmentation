# Import required packages

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Reading the data frame

data = pd.read_csv('C:/Users/Sai Praneeth S/Desktop/Machine_learning_Project/Customer Segmentation/OnlineRetail.csv',
                   encoding = 'unicode_escape')

data.head()

print('Number of null values in customer_id =', data['CustomerID'].isna().sum())

data.dropna(inplace = True)
data.isna().sum()
data.info()

# Create total amount column
data['Total_Amount'] = data['Quantity']*data['UnitPrice']
data.head()
# Data Modelling

# Total amount of transactions
# This data shows how much each customer bought
grouped_data = data.groupby('CustomerID')['Total_Amount'].sum()
grouped_data = grouped_data.reset_index()
grouped_data.head()

# Number of transcations can be calculated by inovices
frequent_data = data.groupby('CustomerID')['InvoiceNo'].count()
frequent_data = frequent_data.reset_index()
frequent_data.columns = ['CustomerID','Frequency']

# To find the last purchase of a customer

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'],
                                     format = '%d-%m-%Y %H:%M')
last_day = max(data['InvoiceDate'])

# To figure out when did the customer purchase from the laste date of data
data['difference'] = last_day - data['InvoiceDate']
# to convert each timedelta to output only number of days
def get_days(x):
    y = str(x).split()[0]
    return int(y)
data['difference'] = data['difference'].apply(get_days)
recent_purchase = data.groupby('CustomerID')['difference'].min()
recent_purchase = recent_purchase.reset_index()

# grouping data for analysis

grouped_df = pd.merge(grouped_data, frequent_data, on = 'CustomerID',how = 'inner')
RFM_df = pd.merge(grouped_df, recent_purchase, on ='CustomerID', how = 'inner')
RFM_df.columns = ['CustomerID','Monetary','Frequency','Recency']

rfm_df = RFM_df.drop('CustomerID', axis = 1)

fig,axes = plt.subplots(1,3, figsize=(20,5))
for i, feature in enumerate(rfm_df.columns):
    sns.distplot(rfm_df[feature], ax=axes[i])

display(rfm_df.describe())

sns.heatmap(rfm_df.iloc[:, 0:3].corr())


































