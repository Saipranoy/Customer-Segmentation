
# Import required packages

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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

# grouping data for analysis

grouped_df = pd.merge(grouped_data, frequent_data, on = 'CustomerID',how = 'inner')
RFM_df = pd.merge(grouped_df, recent_purchase, on ='CustomerID', how = 'inner')
RFM_df.columns = ['CustomerID','Monetary','Frequency','Recency']

# Detecting Outliers
# As K-means clustering access every data point to form a cluster
# So the outliers can affect in the formation of clusters, its better to remove outliers in this scenario

# lets plot box plot of each column


plt.boxplot(RFM_df['Frequency'])
plt.show()

plt.boxplot(RFM_df['Recency'])
plt.show()

plt.boxplot(RFM_df['Monetary'])
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

# Modelling 

k_values = list(range(1,10))
wcss_list = []

for k in k_values:
    kmeans = KMeans(n_clusters = k)
    kmeans.fit_transform(rfm_df_scaled)
    wcss_list.append(kmeans.inertia_)

plt.plot(k_values,wcss_list)
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.title("Within Cluster Sum of Squares - by k")
plt.tight_layout()
plt.show()

# final model with k = 3
kmeans = KMeans(n_clusters = 3)
kmeans.fit(rfm_df_scaled)

# Assigning the labels with the data
RFM_df['Cluster no.'] = kmeans.labels_

# Plot the clusters
sns.boxplot(x = 'Cluster no.', y = 'Monetary', data = RFM_df)

sns.barplot(x='Cluster no.', y = 'Monetary', data = RFM_df)
sns.barplot(x='Cluster no.', y = 'Recency', data = RFM_df)
sns.barplot(x='Cluster no.', y = 'Frequency', data = RFM_df)

plt.figure(figsize = (15,5))
sns.scatterplot(x = RFM_df['Monetary'], 
                y = RFM_df['Frequency'],
                hue = RFM_df['Cluster no.'],
                palette= sns.color_palette('hls',3))
plt.show()

# Lets look at the plot of clusters
clusters = RFM_df.groupby('Cluster no.')
centroids = kmeans.cluster_centers_

for cluster,data in clusters:
    plt.scatter(data["Monetary"],data["Frequency"],data['Recency'],marker = "o",label = cluster)
    plt.scatter(centroids[cluster,0],centroids[cluster,1],marker = "X", color = "Black",s=300)
plt.legend()
plt.tight_layout()
plt.show()

# As we see its not perfectly grouped lets do feature selection and see the result

pca = PCA(n_components = None)
pca.fit(RFM_df)

# Extract the expected variances across components

explained_variance = pca.explained_variance_ratio_
explained_variance_cumulative = pca.explained_variance_ratio_.cumsum()


pca = PCA(n_components = 3)
components = pca.fit_transform(RFM_df)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color = 'yellow')
plt.xlabel('PCA componets')
plt.ylabel('Variance %')
plt.xticks(features)
plt.tight_layout()
plt.show()

pca_components = pd.DataFrame(components)
# Lets build the model based on 2 components

k_values = list(range(1,10))
wcss_list = []

for k in k_values:
    kmeans = KMeans(n_clusters = k)
    kmeans.fit_transform(pca_components.iloc[:,:2])
    wcss_list.append(kmeans.inertia_)

plt.plot(k_values,wcss_list, '-o',color = 'red')
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.title("Within Cluster Sum of Squares - by k")
plt.tight_layout()
plt.show()

# it looks like again the optimal number of clusters is 3
model = KMeans(n_clusters = 3)

clusters = model.fit_predict(pca_components.iloc[:,:2])
RFM_df['Cluster no.'] = clusters

clusters = RFM_df.groupby('Cluster no.')
centroids = kmeans.cluster_centers_

for cluster,data in clusters:
    plt.scatter(data["Monetary"],data["Frequency"],data['Recency'],marker = "o",label = cluster)
    plt.scatter(centroids[cluster,0],centroids[cluster,1],marker = "X", color = "Black",s=300)
plt.legend()
plt.tight_layout()
plt.show()




































