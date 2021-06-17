#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings 
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd  
warnings.filterwarnings('ignore') 
from matplotlib.cm import rainbow
 
# Other libraries 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
 
# Machine Learning 
from sklearn.svm import SVC 
 
dataset = pd.read_csv(r"D:/Heart Disease/heart.csv") 
svc_scores = [] 
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
y = dataset['target'] 
X = dataset.drop(['target'], axis = 1) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

for i in range(len(kernels)):     
    svc_classifier = SVC(kernel = kernels[i])     
    svc_classifier.fit(X_train, y_train) 
    svc_scores.append(svc_classifier.score(X_test, y_test)) 
colors = rainbow(np.linspace(0, 1, len(kernels))) 
plt.bar(kernels, svc_scores, color = colors) 
for i in range(len(kernels)):     
	plt.text(i, svc_scores[i], svc_scores[i]) 
plt.xlabel('Kernels') 
plt.ylabel('Scores') 
plt.title('Support Vector Classifier scores for different kernels') 
plt.show()


# In[ ]:




