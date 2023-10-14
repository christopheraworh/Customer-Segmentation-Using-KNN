#!/usr/bin/env python
# coding: utf-8

# In[23]:


#Loading our necessary packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')



# In[24]:


#Loading our data
df = pd.read_csv('Mall_Customers.csv')

df.columns = ['ID', 'Gender', 'Age', 'AnnualIncome','SpendingScore']


# In[25]:


df.head()


# In[ ]:





# # Univariate Analysis

# In[26]:


df.describe()


# In[27]:


checking_distn = ['Age','AnnualIncome','SpendingScore']

for i in checking_distn:
    plt.figure()
    sns.distplot(df[i])
    plt.title(f'Distribution of {i} feature\n')
    plt.show()


# In[28]:


for i in checking_distn:
    plt.figure()
    sns.boxplot(data=df, x='Gender', y= df[i])
    plt.title(f'Checking for Outlier for {i} using a Boxplot')
    plt.show()


# In[29]:


df['Gender'].value_counts(normalize = True)


# 
# <h1 style="background-color: black; color: white">Bivariate Analysis</h1>
# 

# In[30]:


sns.scatterplot(data= df, x = 'AnnualIncome', y = 'SpendingScore')
plt.show()


# In[31]:


sns.pairplot(df, hue ='Gender')


# In[32]:


df.groupby(['Gender'])['Age', 'AnnualIncome', 'SpendingScore'].mean()


# In[33]:


# Checking correlation across teh variables
corr = df.corr()

sns.heatmap(corr, annot = True,
           cmap = 'plasma', linecolor ='white',
           linewidth=1.7)

plt.title('Correlation Analysis across Features\n')
plt.xticks(rotation=90)
plt.show()


# 
# <h1 style="background-color: black; color: white">Clustering Analysis using KNN</h1>
# 

# In[34]:


clustering1 = KMeans(n_clusters=10) # by default this is 8

clustering1.fit(df[['AnnualIncome']])


# In[35]:


clustering1.labels_


# In[36]:


df['IncomeCluster']  = clustering1.labels_


# In[37]:


df['IncomeCluster'].value_counts()


# In[38]:


clustering1.inertia_ # Tells the distance beween the centroid


# In[39]:


df.groupby('IncomeCluster')['Age','AnnualIncome','SpendingScore'].mean()


# In[40]:


clustering2 = KMeans()
clustering2.fit(df[['AnnualIncome', 'SpendingScore']])

df['SpeningandIncomeCluster'] = clustering2.labels_

df.head()


# In[41]:


inertia_score  = []

for i in range(1,17):
    kmeans = KMeans(n_clusters= i)
    kmeans.fit(df[['AnnualIncome','SpendingScore']])
    inertia_score.append(kmeans.inertia_)

    
plt.plot(range(1,17),inertia_score)
plt.show()


# In[45]:


sns.scatterplot(data =df, x='AnnualIncome', y='SpendingScore',hue='SpeningandIncomeCluster')
plt.show()


# In[ ]:




