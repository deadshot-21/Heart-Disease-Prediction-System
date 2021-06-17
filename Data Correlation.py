#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import rcParams 
from matplotlib.cm import rainbow 
 
import warnings 
warnings.filterwarnings('ignore') 
 
# Other libraries 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
 
# Machine Learning 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 

dataset = pd.read_csv(r"D:/Heart Disease/heart.csv")
rcParams['figure.figsize'] = 16,12 
plt.matshow(dataset.corr()) 
plt.yticks(np.arange(dataset.shape[1]), dataset.columns) 
plt.xticks(np.arange(dataset.shape[1]), dataset.columns) 
plt.colorbar() 
dataset.hist() 
 
rcParams['figure.figsize'] = 16,12
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'green']) 
plt.xticks([0, 1]) 
plt.xlabel('Target Classes') 
plt.ylabel('Count') 
plt.title('Count of each Target Class')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




