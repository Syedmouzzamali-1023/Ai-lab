#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data = pd.read_csv('HR_comma_sep.csv')
data.head()


# In[2]:


data.info()


# In[3]:


data['Departments'].value_counts()


# In[4]:


data['Departments'].unique()


# In[5]:


data['salary'].unique()


# In[6]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
print(le)
data['salary']=le.fit_transform(data['salary'])
data['Departments']=le.fit_transform(data['Departments'])


# In[7]:


data['salary'].unique()


# In[8]:


X=data.iloc[:,:-1]
X


# In[9]:


X=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','Departments','salary']]
y=data['left']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=42)


# In[10]:


X_train


# In[11]:


y_train


# In[16]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(
                    hidden_layer_sizes=(6,5),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01,
                   )
clf.fit(X_train,y_train)


# In[17]:


ypred=clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,ypred)


# In[18]:


X_test.shape


# In[19]:


n=pd.DataFrame({
    'satisfaction_level':[0.78],
    'last_evaluation':[0.53],
    'number_project':[2],
    'average_montly_hours':[157],
    'time_spend_company':[3],
    'Work_accident':[0],
    'promotion_last_5years':[0],
    'Departments':[1],
    'salary':[1]
    
    
})


# In[20]:


new_data=clf.predict(n)
print(new_data)


# In[22]:


from sklearn.metrics import classification_report
print(classification_report(y_test, ypred))


# In[23]:


from sklearn.metrics import confusion_matrix
ypred = clf.predict(X_test)
print(confusion_matrix(y_test, ypred))


# In[24]:


print(y_train.value_counts())
print(y_test.value_counts())


# In[ ]:




