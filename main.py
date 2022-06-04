#Imports

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import scipy
import librosa
import librosa.display
import IPython.display as ipd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import pickle

dataf = pd.read_csv("C:/Users/home/OneDrive/Desktop/Datasets/GTZAN Dataset - Music Genres/Data/features_3_sec.csv")
dataf.head()
dataf.tail()

dataf.shape

dataf.describe()

dataf = dataf.drop(labels='filename',axis=1)

class_list = dataf.iloc[:,-1]
convert = LabelEncoder()

y = convert.fit_transform(class_list)
y

dataf.iloc[:,:-1]

from sklearn.preprocessing import StandardScaler
fit = StandardScaler()
X = fit.fit_transform(np.array(dataf.iloc[:,:-1],dtype=float))

X_train,x_test, Y_train, y_test = train_test_split(X,y,test_size=0.2)

print(len(Y_train),len(y_test))

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()

model.add(Dense(512,input_shape=(X_train.shape[1],),activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10,activation='softmax'))


model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='accuracy')

earlystop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10,min_delta=0.0001)
modelcheck = ModelCheckpoint('best_model.hdf5',monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')

history = model.fit(X_train,Y_train, validation_data=(x_test,y_test), epochs=600, callbacks=[earlystop,modelcheck], batch_size=128)

from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

test_loss, test_accuracy = model.evaluate(x_test,y_test,batch_size=128)
print("Test loss : ",test_loss)
print("\nBest test accuracy : ",test_accuracy*100)
