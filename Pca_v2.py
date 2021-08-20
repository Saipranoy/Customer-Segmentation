# Import Required packages

import pandas as pd

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

# Reading the data frame

data = pd.read_csv('C:/Users/Sai Praneeth S/Desktop/Machine_learning_Project/Customer Segmentation/OnlineRetail.csv',
                   encoding = 'unicode_escape')

data.head()

# Missing Values

data.info()

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
recent_purchase.columns = ['CustomerID','Recency']
recent_purchase.head()
# grouping data for analysis

grouped_df = pd.merge(grouped_data, frequent_data, on = 'CustomerID',how = 'inner')
RFM_df = pd.merge(grouped_df, recent_purchase, on ='CustomerID', how = 'inner')
RFM_df.columns = ['CustomerID','Monetary','Frequency','Recency']
RFM_df.head()
# Detecting Outliers
# As K-means clustering access every data point to form a cluster
# So the outliers can affect in the formation of clusters, its better to remove outliers in this scenario

# lets plot box plot of each column


plt.boxplot(RFM_df['Frequency'])
plt.xlabel('Frequency')
plt.show()

plt.boxplot(RFM_df['Recency'])
plt.xlabel('Recency')
plt.show()

plt.boxplot(RFM_df['Monetary'])
plt.xlabel('Monetary')
plt.show()


outlier_vars = ['Monetary','Recency','Frequency']

for column in outlier_vars:
    
    lower_quartile = RFM_df[column].quantile(0.25)
    upper_quartile = RFM_df[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 1.5
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = RFM_df[(RFM_df[column] < min_border) | (RFM_df[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    RFM_df.drop(outliers, inplace = True)
    
# Rescaling the data
rfm_df = RFM_df[['Monetary','Frequency','Recency']]
scale_standardisation = StandardScaler()

rfm_df_scaled = scale_standardisation.fit_transform(rfm_df)

rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['monetary','frequency','recency']
rfm_df_scaled.head()
# Instatntiate pca and apply it on the scaled data frame

pca = PCA(n_components = None)
components = pca.fit(rfm_df_scaled)

# Extract the expected variances across components

explained_variance = components.explained_variance_ratio_
explained_variance_cumulative = components.explained_variance_ratio_.cumsum()

###################################################################

# Plot the explained variances of Components

##################################################################

# Create list for number of components
num_vars_list = list(range(1,4))
plt.figure(figsize=(15,10))

# plot the explained variances of each component
plt.subplot(2,1,1)
plt.bar(num_vars_list,explained_variance)
plt.xlabel("Number of Components")
plt.ylabel("% of variance")
plt.title("Variance across Components")
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(num_vars_list, explained_variance_cumulative)
plt.title("Cumulative variances across components")
plt.xlabel("Cumulative % Variance")
plt.ylabel("Number of Components")
plt.tight_layout()
plt.show()

# Apply pca with selected required components 
pca = PCA(n_components = 0.75)
rfm_data = pca.fit_transform(rfm_df_scaled)

pca.n_components_

rfm_d = pd.DataFrame(rfm_data)
# Apply the components to K-means clustering

k_values = list(range(1,10))
wcss_list = []

for k in k_values:
    kmeans = KMeans(n_clusters = k)
    kmeans.fit_transform(rfm_d.iloc[:,:2])
    wcss_list.append(kmeans.inertia_)

plt.plot(k_values,wcss_list, '-o',color = 'red')
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.title("Within Cluster Sum of Squares - by k")
plt.tight_layout()
plt.show()

# visualise the clusters for the model
k_model = KMeans(n_clusters=3)
clusters = k_model.fit_predict(rfm_d.iloc[:,:2])

RFM = rfm_df_scaled 
RFM['labels'] = clusters 

RFM.head()

clusters = RFM.groupby('labels')
centroids = k_model.cluster_centers_

for cluster,data in clusters:
    plt.scatter(data["monetary"],data["frequency"],data['recency'],marker = "o",label = cluster)
    #plt.scatter(centroids[cluster,0],centroids[cluster,1],marker = "X", color = "Black",s=300)
plt.legend()
plt.tight_layout()
plt.show()

# 3d visualisations
fig = plt.figure(figsize=(21,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(RFM["monetary"][RFM.labels == 0], RFM["frequency"][RFM.labels == 0], RFM["recency"][RFM.labels == 0], c='blue', s=60)
ax.scatter(RFM["monetary"][RFM.labels == 1],RFM["frequency"][RFM.labels == 1], RFM["recency"][RFM.labels == 1], c='red', s=60)
ax.scatter(RFM["monetary"][RFM.labels == 2], RFM["frequency"][RFM.labels == 2], RFM["recency"][RFM.labels == 2], c='yellow', s=60)

ax.view_init(30, 185)
plt.show()

# Cluster analysis

data = data = pd.read_csv('C:/Users/Sai Praneeth S/Desktop/Machine_learning_Project/Customer Segmentation/OnlineRetail.csv',
                   encoding = 'unicode_escape')

RFM_df['label'] = k_model.predict(rfm_d.iloc[:,:2])

sns.barplot(x = 'Monetary', y='label',data=RFM_df)

RFM.head()

##########################################
# Cluster Analysis
##########################################


rfm_df['Clusters'] = k_model.labels_

analysis = rfm_df.groupby('Clusters').agg({
    'Recency':['mean','max','min'],
    'Frequency':['mean','max','min'],
    'Monetary':['mean','max','min','count']})

# '0' cluster is best and '1' cluster is better to be avoided.










