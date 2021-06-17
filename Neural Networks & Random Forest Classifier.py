#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
import keras 
from keras.models import Sequential 
from keras.layers import Dense 

import warnings 
import numpy as np 
import pandas as pd
 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report,confusion_matrix 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import average_precision_score 
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score 

import matplotlib.cm as lm 

df = pd.read_csv("D:/Heart Disease/heart.csv") 
df.head(5) 
chest_pain=pd.get_dummies(df['cp'],prefix='cp',drop_first=True) 
df=pd.concat([df,chest_pain],axis=1) 
df.drop(['cp'],axis=1,inplace=True) 
sp=pd.get_dummies(df['slope'],prefix='slope') 
th=pd.get_dummies(df['thal'],prefix='thal') 
rest_ecg=pd.get_dummies(df['restecg'],prefix='restecg') 
frames=[df,sp,th,rest_ecg] 
df=pd.concat(frames,axis=1) 
df.drop(['slope','thal','restecg'],axis=1,inplace=True) 
X = df.drop(['target'], axis = 1) 
y = df.target.values 

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 
classifier = Sequential() 

# input layer 
classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 22)) 

#hidden layer 
classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu')) 

# output layer 
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50) 

# Predicting the Test set results 
y_pred = classifier.predict(X_test) 
# import seaborn as sns 
# from sklearn.metrics import confusion_matrix 
# cm = confusion_matrix(y_test, y_pred.round()) 

# sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False) 

#accuracy score 
from sklearn.metrics import accuracy_score 
ac=accuracy_score(y_test, y_pred.round())
print() 
print('Neural Network Accuracy: ',ac)
print()

 
rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0) 
rdf_c.fit(X_train,y_train) 
rdf_pred=rdf_c.predict(X_test) 
rdf_cm=confusion_matrix(y_test,rdf_pred) 
rdf_ac=accuracy_score(rdf_pred,y_test) 

# plt.title("rdf_cm") 
# sns.heatmap(rdf_cm,annot=True,fmt="d",cbar=False) 
print('Random Forest Accuracy:',rdf_ac) 
rf_scores = [] 
estimators = [10, 100, 200, 500, 1000] 
for i in estimators: 
	rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0) 
	rf_classifier.fit(X_train, y_train) 
	rf_scores.append(rf_classifier.score(X_test, y_test)) 
estimators = [10, 100, 200, 500, 1000] 
colors = lm.rainbow(np.linspace(0, 1, len(estimators))) 
plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8) 

for i in range(len(estimators)): 
	plt.text(i, rf_scores[i], rf_scores[i]) 
plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators]) 
plt.xlabel('Number of estimators') 
plt.ylabel('Scores') 
plt.title('Random Forest Classifier scores for different number of estimators') 
plt.show() 

def plotting(true,pred): 
	fig,ax=plt.subplots(1,2,figsize=(10,5)) 
	precision,recall,threshold = precision_recall_curve(true,pred[:,1]) 
	ax[0].plot(recall,precision,'g--')
	ax[0].set_xlabel('Recall') 
	ax[0].set_ylabel('Precision') 
	ax[0].set_title("Average Precision Score : {}".format(average_precision_score(true,pred[:,1]))) 
	fpr,tpr,threshold = roc_curve(true,pred[:,1]) 
	ax[1].plot(fpr,tpr) 
	ax[1].set_title("AUC Score is: {}".format(auc(fpr,tpr))) 
	ax[1].plot([0,1],[0,1],'k--') 
	ax[1].set_xlabel('False Positive Rate') 
	ax[1].set_ylabel('True Positive Rate') 
	
plotting(y_test,rdf_c.predict_proba(X_test)) 
#plt.figure() 
plt.show() 

model_accuracy = pd.Series(data=[rdf_ac,ac], index=['RandomForest','Neural Network']) 
fig= plt.figure(figsize=(8,8)) 
model_accuracy.sort_values().plot.barh() 
plt.title('Model Accracy') 
plt.show()


# In[ ]:




