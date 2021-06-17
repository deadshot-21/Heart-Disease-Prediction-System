#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier 
dataset = pd.read_csv("D:/Heart Disease/heart.csv") 
y = dataset['target'] 
X = dataset.drop(['target'], axis = 1) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0) 
 
dt_scores = [] 
for i in range(1, len(X.columns) + 1):     
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)     
    dt_classifier.fit(X_train, y_train) 
    dt_scores.append(dt_classifier.score(X_test, y_test)) 
plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green') 
for i in range(1, len(X.columns) + 1): 
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1])) 
plt.xticks([i for i in range(1, len(X.columns) + 1)]) 
plt.xlabel('Max features') 
plt.ylabel('Scores') 
plt.title('Decision Tree Classifier scores for different number of maximum features') 
plt.show()


# In[ ]:




