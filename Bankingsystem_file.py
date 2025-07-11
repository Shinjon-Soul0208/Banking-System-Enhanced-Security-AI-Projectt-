# Banking-System-Enhanced-Security-AI-Project-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as map
#imports have been made
data=pd.read_csv("/content//Credit Card Customer Data.csv")
#Extract the independent and dependent value:
x=data.iloc[:,[3,4]].values
#finding optimal number of clusters using elbow method :
"""For determining  K(numbers of clusters) we use Elbow method.
Elbow Method is a technique that we use to determine the number of centroids(k) to use in a k-means clustering algorithm.
In this method to determine the k-value we continuously iterate for k=1 to k=n (Here n is the hyperparameter that we choose as per our requirement).
For every value of k, we calculate the within-cluster sum of squares (WCSS) value.
***WCSS[WITHIN CLUSTERS SUM OF SQUARES ] - It is defined as the sum of square distances between the centroids and
each points***
"""
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#STEP1 HAS BEEN PROCCESSED .


from sklearn.cluster import KMeans
# Initialize KMeans model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
# Fit the model to the data and predict cluster labels:
y_predict = kmeans.fit_predict(x)

#visulaizing the clusters
plt.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster
plt.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster
plt.scatter(x[y_predict== 2, 0], x[y_predict == 2, 1], s = 100, c = 'red', label = 'Cluster 3') #for third cluster
plt.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') #for fourth cluster
plt.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') #for fifth cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')
plt.title('Clusters of customers')
plt.xlabel('Credit Crad limit (Rs.):)')
plt.ylabel('Total Credit Crads :)')
plt.legend()
plt.show()

#CLUSTERING OF THE DATA SET :
import pandas as pd
import numpy as mp
import matplotlib.pyplot as plt
import seaborn as sns
import  plotly.express as px
#import ploty.graph_objects as go
from plotly.subplots import make_subplots
#imports have been made :

#The dataset has been inluded in a daatframe using pandas :
data = pd.read_csv("/content//Credit Card Customer Data.csv")
data.head()
data.describe()

#dataset count of rows and columns :
data.shape

data.info()

data.duplicated().sum()

# We can use a heatmap to check correlation between the variables.
corr = data.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr,cbar=True,square=True,fmt='.1f',annot=True,cmap='Reds')

data.columns

data.describe(include='all')

for items in data.columns.tolist():
  print(items)
  print("The number of unique values in columns :",data[items].unique())

#DATA WRANGLING :
data=data.apply(pd.to_numeric,errors='coerce')
data.info()

data=data.drop(['Avg_Credit_Limit'],axis=1)
data.head()

data['Total_visits_bank'].value_counts()

data['Total_visits_online'].value_counts()

data['Total_calls_made'].value_counts()

#DATA VISUALIZATION:
numeric_features = data.describe().columns
numeric_features

for col in numeric_features[:]:
  fig=plt.figure(figsize=(10,6))
  feature=data[col]
  ax=fig.gca()
  feature.hist(ax=ax)
  ax.axvline(feature.mean(),color='red',linestyle='dashed',linewidth=2)
  ax.axvline(feature.median(),color='green',linestyle='dashed',linewidth=2)
  ax.set_title(col)
plt.show()


data1=data[['Total_Credit_Cards','Customer Key']]
data2=data1.groupby(['Customer Key'],as_index = False).aggregate('sum')
total_credit_cards_sum=data1['Total_Credit_Cards'].aggregate('sum')
l=list(data2['Total_Credit_Cards']*(100/total_credit_cards_sum))
l=l[:10]
print(l)

customers=data2['Customer Key'].astype(str).transpose().tolist()
customer=customers[:10]
print(customers)

# Calculate the number of elements to take, based on the shorter list
num_elements = min(len(data2['Customer Key']), len(l))
# Then, slice both lists using this calculated length
customers = data2['Customer Key'].astype(str).tolist()[:num_elements]
l = l[:num_elements]
fig = plt.figure(figsize=(10, 7)) # Reduced figure size for better display
plt.pie(l, labels=customers)
plt.show()

#END OF CODE.
