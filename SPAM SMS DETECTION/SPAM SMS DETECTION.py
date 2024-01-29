#!/usr/bin/env python
# coding: utf-8

# In[248]:


# import necessory Libraries

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# In[249]:


# load a spam  dataset 

data=pd.read_csv("H:\CodIt Solutions\SMS Spam Dataset/spam.csv", index_col=False)
data.head()


# In[250]:


# remove extra column "Unnamed: 0"

data.drop("Unnamed: 0", axis=1, inplace=True)
data.head()


# In[251]:


# we see random data and lenth of data

x=data['Message'][143]
print(len(x))
print(data['Message'][143])


# In[252]:


# convert a string into categorical value

LE=LabelEncoder()
y=LE.fit_transform(data['spamORham'])


# In[253]:


# check how many ham and spam there in the dataset
# class are imbalance 

data['spamORham'].value_counts()


# In[254]:


main_data=pd.concat([data['Message'], pd.Series(y)], axis=1, keys=['Message', 'spamORham'])


# In[255]:


main_data.head()


# In[256]:


#randomly take a ham in the dataset

ham=main_data[main_data['spamORham']==0].sample(n=747)
spam=main_data[main_data['spamORham']==1].sample(n=747)


# In[257]:


sortdata=pd.concat([ham,spam], axis=0)


# In[258]:


sortdata


# In[259]:


voc_size=1000000


# In[260]:


#covert a word into vector using WordEmbedding

one_rep=[one_hot(words,voc_size)for words in sortdata.Message]


# In[261]:


print(one_rep)


# In[262]:


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[263]:


sent_length=150
emb_sequence=pad_sequences(one_rep, padding='pre', maxlen=sent_length)
print(emb_sequence)


# In[264]:


X_train, X_test, y_train, y_test=train_test_split(emb_sequence, sortdata['spamORham'], test_size=0.1, random_state=42)


# In[265]:


from sklearn.ensemble import RandomForestClassifier


# In[266]:


clf = RandomForestClassifier(max_depth=2, random_state=42)


# In[267]:


clf.fit(X_train, y_train)


# In[268]:


y_pred = model.predict(X_test)

print(y_test[:20])
print(y_pred[:20])


# In[269]:


from sklearn.metrics import accuracy_score, classification_report


# In[270]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[271]:


# Display the classification report
print("\nClassification Info:")
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




