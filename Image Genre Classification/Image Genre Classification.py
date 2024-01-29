#!/usr/bin/env python
# coding: utf-8

# In[463]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[464]:


traindata = pd.read_csv(r"H:\CodIt Solutions\Movie Genre Classification\archive (6)/Movie_combined.csv", index_col=False)


# In[465]:


traindata.drop("Unnamed: 0", axis=1, inplace=True)


# In[466]:


traindata.head()


# In[467]:


traindata.Genre.value_counts()


# In[468]:


LE=LabelEncoder()
y=LE.fit_transform(traindata['Genre'])


# In[469]:


X=traindata.Plot
y=y
X_ = pd.concat([traindata['Plot'], pd.Series(y)], axis=1, keys=['Plot', 'Genre'])

X_.head()

X_['Genre'].value_counts()


# In[470]:


X_.head()


zero=X_[X_['Genre']==0].sample(n=602)
one=X_[X_['Genre']==1].sample(n=602)
two=X_[X_['Genre']==2].sample(n=602)
three=X_[X_['Genre']==3].sample(n=602)
four=X_[X_['Genre']==4].sample(n=602)
five=X_[X_['Genre']==5].sample(n=602)


# In[471]:


X_=pd.concat([zero,one,two,three,four,five], axis=0)


# In[472]:


X_.head()


# In[473]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming X_ is a DataFrame with a 'Genre' column containing lists of genres for each example
# If 'Genre' is a string with comma-separated genres, you may need to split it into a list first
# Example: X_['Genre'] = X_['Genre'].apply(lambda x: x.split(','))

# Create a DataFrame with one-hot encoding for each genre
genre_df = pd.get_dummies(X_['Genre'].explode()).groupby(level=0).sum()

# Plot the bar chart
plt.figure(figsize=(5, 5))
genre_df.sum().sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Undersampling of Genres')
plt.xlabel('Genre')
plt.ylabel('Frequency')
plt.xticks(rotation=0, ha='right')
plt.show()


# In[486]:


def custom_tokenizer(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return words


# In[487]:


ex_TfidfVectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, stop_words='english', max_features=5000, lowercase=True)

# Fit and transform the 'Genre' column
tfidf_matrix = ex_TfidfVectorizer.fit_transform(X_['Plot'])


# In[476]:


print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())
print(tfidf_matrix.shape)

print(ex_TfidfVectorizer.vocabulary_)


# In[477]:


ex_TfidfVectorizer.get_feature_names_out()[:20]


# In[ ]:





# In[478]:


X_train, X_test, y_train, y_test=train_test_split(tfidf_matrix, X_['Genre'], test_size=0.1, random_state=42)


# In[479]:


model = LogisticRegression(max_iter=300)


# In[480]:


from sklearn.model_selection import cross_val_score
k = 5
cv_scores = cross_val_score(model, tfidf_matrix, X_['Genre'], cv=k, scoring='accuracy')


# In[481]:


model.fit(X_train, y_train)


# In[482]:


y_pred = model.predict(X_test)

print(y_test[:20])
print(y_pred[:20])


# In[483]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[484]:


# Display the classification report
print("\nClassification Info:")
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




