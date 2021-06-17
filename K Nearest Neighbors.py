#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings 
 
from matplotlib import pyplot as plt 
import pandas as pd 
 
warnings.filterwarnings('ignore') 
 
# Other libraries 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
 
# Machine Learning 
from sklearn.neighbors import KNeighborsClassifier 
 
dataset = pd.read_csv(r"D:/Heart Disease/heart.csv") 
dataset.info() 
dataset.describe() 
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']) 
standardScaler = StandardScaler() 
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] 
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale]) 
y = dataset['target'] 
X = dataset.drop(['target'], axis = 1) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0) 
knn_scores = [] 
for k in range(1,21):   
	  knn_classifier = KNeighborsClassifier(n_neighbors = k)   
	  knn_classifier.fit(X_train, y_train)     
	  knn_scores.append(knn_classifier.score(X_test, y_test)) 
plt.plot([k for k in range(1, 21)], knn_scores, color = 'red') 
for i in range(1,21):   
	  plt.text(i, knn_scores[i-1], (i, knn_scores[i-1])) 
plt.xticks([i for i in range(1, 21)]) 
plt.xlabel('Number of Neighbors (K)') 
plt.ylabel('Scores') 
plt.title('K Neighbors Classifier scores for different K values')
plt.show() 


# In[ ]:




