
# coding: utf-8

# In[1]:




#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.externals import joblib

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve , auc , roc_auc_score
from matplotlib import pyplot
from sklearn.naive_bayes import GaussianNB
cm1=[]

cm2=[]
cm3=[]
# In[2]:


#importing the dataset
dataset = pd.read_csv("dataset/phishcoop.csv")
dataset = dataset.drop('id', 1) #removing unwanted column
x = dataset.iloc[ : , :-1].values
y = dataset.iloc[:, -1:].values


# In[3]:


#spliting the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state =1 )


# # GaussianNB

# In[14]:


NB_classifier = GaussianNB()


# In[15]:


NB_classifier.fit(x_train, y_train)
#predicting the tests set result
y_pred = NB_classifier.predict(x_test)
#confusion matrix
cm1 = confusion_matrix(y_test, y_pred)


# # Support Vector Machine

# In[58]:


#applying grid search to find best performing parameters 
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[0.1,0.5,1,5,10], 'gamma': [ 0.1, 0.2,0.3, 0.5 , 0.6 , 0.7]}]
#grid_search = GridSearchCV(SVC(kernel='rbf' ),  parameters,cv =5, n_jobs= -1)
#grid_search.fit(x_train, y_train)

#printing best parameters 
#print("Best Accurancy =" +str( grid_search.best_score_))
#print("best parameters =" + str(grid_search.best_params_)) 


# In[82]:


#fitting kernel SVM  with best parameters calculated 

svm_classifier = SVC(C=5, kernel = 'rbf', gamma = 0.2 , random_state = 0 , class_weight='balanced' , probability=True )
svm_classifier.fit(x_train, y_train)

#predicting the tests set result
y_pred = svm_classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred)

# In[83]:


svm_classifier.score(x_train, y_train)


# In[84]:


svm_classifier.score(x_test, y_test)


# In[85]:


# roc curve and auc
from sklearn.metrics import roc_curve , roc_auc_score
from matplotlib import pyplot
probs = svm_classifier.predict_proba(x_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
        #print('AUC: %.3f' % auc)
# calculate roc curve
         #fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
        #pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
        #pyplot.plot(fpr, tpr, marker='.')
# show the plot
         #pyplot.show()


# # ELM Model

# In[48]:


from sklearn_extensions.extreme_learning_machines.elm import ELMClassifier


# In[54]:


elm_classifier = ELMClassifier(n_hidden=1000 ,activation_func='tanh' ,alpha=0.9 , random_state=1 )


# In[50]:


elm_classifier.fit(x_train, y_train)


# In[51]:


#predicting the tests set result
y_pred = elm_classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred)

# In[52]:


elm_classifier.score(x_train, y_train)


# In[53]:


elm_classifier.score(x_test, y_test)



print("NB")
print ("Accuraccy" , (float(cm1[0][0])/(float(cm1[0][0])+float(cm1[1][0])))*100)
print ("Specificity" , float(cm1[1][1])/(float(cm1[0][1])+float(cm1[1][1])))
print(cm1)


print("SVM")
print ("Accuraccy" , (float(cm2[0][0])/(float(cm2[0][0])+float(cm2[1][0])))*100)
print ("Specificity" , float(cm2[1][1])/(float(cm2[0][1])+float(cm2[1][1])))
print(cm2)


print("ELM")
print ("Accuraccy" , (float(cm3[0][0])/(float(cm3[0][0])+float(cm3[1][0])))*100)
print ("Specificity" , float(cm3[1][1])/(float(cm3[0][1])+float(cm3[1][1])))
print(cm3)


import matplotlib.pyplot as plt 
  
# x-coordinates of left sides of bars  
left = [1, 2, 3] 
  
# heights of bars 
height = [(float(cm1[0][0])/(float(cm1[0][0])+float(cm1[1][0])))*100, (float(cm2[0][0])/(float(cm2[0][0])+float(cm2[1][0])))*100, (float(cm3[0][0])/(float(cm3[0][0])+float(cm3[1][0])))*100] 
  
# labels for bars 
tick_label = ['NB', 'SVM', 'ELM'] 
  
# plotting a bar chart 
plt.bar(left, height, tick_label = tick_label, 
        width = 0.8, color = ['red', 'green','blue']) 
  
# naming the x-axis 
plt.xlabel('x - axis') 
# naming the y-axis 
plt.ylabel('y - axis') 
# plot title 
plt.title('Accuraccy Visualization!') 
  
# function to show the plot 
plt.show()





