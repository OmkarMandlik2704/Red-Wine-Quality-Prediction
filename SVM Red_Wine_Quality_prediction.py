#!/usr/bin/env python
# coding: utf-8

# In[43]:


import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


# In[44]:


df=pd.read_csv('winequality-red.csv')
df.head()


# In[45]:


df.shape


# In[46]:


df.info()


# In[47]:


df.describe()


# In[48]:


#preprocessing
bins=(2,6.5,8)
group_names=['Bad','Good']
categories=pd.cut(df['quality'],bins,labels=group_names)
df['quality']=categories


# In[49]:


df['quality'].value_counts()


# In[50]:


sn.barplot(x='quality',y='alcohol',data=df)


# In[51]:


sn.barplot(x='quality',y='volatile acidity',data=df)


# In[52]:


X=df.drop(['quality'],axis=1)
y=df['quality']


# In[53]:


from sklearn.preprocessing import LabelEncoder
# Encode Target Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)  # 'Good' -> 1, 'Bad' -> 0


# In[54]:


# Convert Categorical Features to Numeric (if any)
for col in df.columns:
    if df[col].dtype == 'object':  # Check if column is categorical
        df[col] = LabelEncoder().fit_transform(df[col])


# In[55]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=45)


# In[56]:


X_test


# In[57]:


y_test


# In[59]:


from sklearn.svm import SVC
my_model=SVC(kernel='rbf',random_state=0)
result=my_model.fit(X_train, y_train)


# In[60]:


predictions=result.predict(X_test)
predictions


# In[61]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predictions)
sn.heatmap(cm,annot=True,fmt='2.0f')


# In[62]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test,predictions))


# In[66]:


new_pred=list(result.predict([[11.1,0.100,0.99,4,0.99,1,2,0.1,1,0.50,9]]))
new_pred


# In[ ]:




