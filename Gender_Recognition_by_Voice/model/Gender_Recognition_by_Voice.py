
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import pprint as pp
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

from keras.callbacks import History 

#read dataset
dataset = pd.read_csv('voice.csv', header=0)

#Split features and labels
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values


print(dataset.isnull().sum())

# fill missing values with mean column values
imputer = Imputer(strategy="mean", axis=0)
transformed_values = imputer.fit_transform(x)

# count the number of NaN values in each column
print(np.isnan(transformed_values).sum())

#Preprocessess data
scaler = StandardScaler()
x = scaler.fit_transform(x)

#Encoding
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y) #Male = 1 ----- Female = 0 
y = to_categorical(y, 2)

#Test-train split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123456)
    
print ("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)


# In[2]:


#Build Neural Network 

n_cols = x_train.shape[1]
n_classes = 2

model = Sequential()
model.add(Dense(300, activation='relu', input_dim = n_cols)) 
model.add(Dropout(0.3))

model.add(Dense(300, activation='relu')) 
model.add(Dropout(0.3))

model.add(Dense(300, activation='relu')) 
model.add(Dropout(0.3))

model.add(Dense(300, activation='relu')) 
model.add(Dropout(0.3))

model.add(Dense(2, activation='softmax'))


# In[3]:


hist = History()
training_epochs = 10
batch_size =40


# In[4]:


# select Cost function and Optimizer
adam = keras.optimizers.Adam(lr=0.0005)
model.compile(optimizer = adam, loss='categorical_crossentropy', metrics=['accuracy'])


# model training
model.fit(x_train, y_train, batch_size = batch_size, epochs = training_epochs, validation_split = .1, callbacks = [hist])


# In[5]:


# Plot loss and validate loss
plt.plot(hist.history['loss'], color = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], color = 'blue', label = 'val-loss')
plt.legend(loc='upper left')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.show()


# In[6]:


# Plot accuracy and validate accuracy
plt.plot(hist.history['acc'], color = 'red', label = 'acc')
plt.plot(hist.history['val_acc'], color = 'blue', label = 'val-acc')
plt.legend(loc='upper left')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.show()


# In[7]:


# Model evaluation using test set
y_pred =  model.predict(x_test)
y_pred = np.round(y_pred)

print('test accuracy:')
print(metrics.accuracy_score(y_pred,y_test))

target_names = ['female', 'male']
print(classification_report(y_test, y_pred, target_names=target_names))

