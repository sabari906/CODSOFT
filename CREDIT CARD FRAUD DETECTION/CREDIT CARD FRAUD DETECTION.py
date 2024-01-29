#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


# In[93]:


data=pd.read_csv(r"H:\CodIt Solutions\Credit Card Fraud Detection/creditcard.csv")
data.head()


# In[94]:


data.info()


# In[95]:


data['Class'].value_counts()


# In[96]:


zero=data[data['Class']==0]
one=data[data['Class']==1]


# In[97]:


zero.Amount.describe() 


# In[98]:


one.Amount.describe()


# In[99]:


data_x=zero.sample(n=492)


# In[100]:


main_data=pd.concat([data_x, one], axis=0)


# In[101]:


main_data.Class.value_counts()


# In[102]:


X=main_data.iloc[:,:-1]
y=main_data.iloc[:,-1]


# In[103]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42) 


# In[ ]:





# In[104]:


model=DecisionTreeClassifier(random_state=42)


# In[105]:


model.fit(X_train, y_train)


# In[106]:


y_pred = model.predict(X_test)


# In[107]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Of Testing: {accuracy:.2f}")


# In[108]:


# Display the classification report
print("\nClassification Info:")
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




