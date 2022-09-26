#!/usr/bin/env python
# coding: utf-8

# # Diabetes prediction

# In[30]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# # Loading the dataset and data Exploration

# In[31]:


diabetes_dataset = pd.read_csv('C:/Users/LEKHITHA/Desktop/Diabeties.csv') 
diabetes_dataset.head()


# In[4]:


diabetes_dataset.max()


# In[5]:


diabetes_dataset = pd.read_csv('C:/Users/LEKHITHA/Desktop/Diabeties.csv') 
diabetes_dataset.tail()


# # Data Processing

# In[6]:


diabetes_dataset.isnull().sum()


# In[7]:


r=diabetes_dataset.shape
print(r)
diabetes_dataset.describe()


# In[8]:


diabetes_dataset.corr()


# # Data Visualization 

# In[9]:


import matplotlib.pyplot as plt
plt.scatter(diabetes_dataset['Age'],diabetes_dataset['Outcome'],color="r")
plt.xlabel('Age')
plt.ylabel('Outcome')
plt.colorbar()
plt.show()


# In[10]:


diabetes_dataset['Outcome'].value_counts()
#0-->for non-diabetic
#1--. for diabetic


# In[11]:


diabetes_dataset.groupby('Outcome').mean()


# # separating the data and labels 

# In[12]:


X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)


# # data standardization

# In[13]:


scaler = StandardScaler()
scaler.fit(X)


# In[14]:


standardized_data = scaler.transform(X)
print(standardized_data)


# In[15]:


X = standardized_data
Y = diabetes_dataset['Outcome']
print(X)
print(Y)


# # train and test

# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


# In[17]:


for k in ('linear', 'poly', 'rbf', 'sigmoid'):
    model = svm.SVC(kernel=k)
    model.fit(X_train, Y_train)
    ypred = model.predict(X_train)
    print(k)
    print(accuracy_score(Y_train, ypred))
classifier = svm.SVC(kernel='rbf')
#training the support vector
classifier.fit(X_train, Y_train)


# # model evaluation

# In[18]:


#accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

#accuracy score on the test day
from sklearn.metrics import accuracy_score
ypred= classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test,ypred)
print('Accuracy score of the test data : ', test_data_accuracy)


# #  making a predictive analysis

# In[19]:


input_data = (1,85,66,29,0,26.6,0.351,31)
input_data_as_numpy_array = np.array(input_data)
# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[20]:


input_data = (10,139,90,0,0,27.1,1.441,57)
input_data_as_numpy_array = np.array(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[21]:


diabetes_dataset.query('Glucose>140 and Outcome==0')


# In[22]:


df1=diabetes_dataset.query(' Outcome==1')
df1


# In[23]:


df2=diabetes_dataset.query('Glucose<140  and Outcome==1 ')
df2


# In[25]:


d=diabetes_dataset.query('BloodPressure<140 and Outcome==1')
d


# In[35]:


d=diabetes_dataset.query('Glucose>126 and Outcome==1')
d


# In[36]:


d=diabetes_dataset.query('BloodPressure<140 and Outcome==1')
d


# In[33]:


d=diabetes_dataset.query('Outcome==0')
d.mean()


# In[43]:


d=diabetes_dataset.query('Outcome==1 and  BloodPressure<140 and Glucose>126')
d


# In[ ]:


BloodPressure : below  140
Glucose >126
Insulin >200mg
BMI   19-23 kg


# In[27]:


diabetes_dataset.mean()


# In[28]:


diabetes_dataset.corr()


# In[29]:


plt.scatter(diabetes_dataset['BloodPressure'] , diabetes_dataset['Glucose'])
plt.xlabel('BloodPressure')
plt.ylabel('Glucose')
plt.title('plot relation between Glucose and BloodPressure')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


diabetes_dataset.query("Pregnancies==0 and Outcome==1")


# In[25]:


Our precision, recall, and f1-score are approximately 0.71, 0.52, and 0.60 respectively.
 The model is not too good, For a healthcare problem, we could end up misdiagnosing patients that have diabetes. 
This is why we pay more attention to the recall score.
 We can improve our results by collecting more data

