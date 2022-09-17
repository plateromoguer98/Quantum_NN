# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:15:21 2022

@author: migui
"""
###### MODEL 1 ##########
import pandas as pd 
import numpy as np
path="/home/carlos/red_neuronal_dipolos/miguel/10000_fotos/input_neural_network_pol_distances_big.dat"
df = pd.read_csv(path, sep=",",engine="python")
the_output=[]
devuelvo=[]
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics 
X= df.iloc[:,0:20]
y=df.iloc[:,21]
print("The y is:",y)
print("The shape of y is",y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


import tensorflow as tf    
from tensorflow import keras 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
import random
norm = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)

model = keras.Sequential()
kerner_regularizer_l1 = regularizers.l1(1e-5)

# Capas ocultas
model.add(layers.Dense(300, input_shape=(20,),activation="tanh", name="Hidden_Layer_1"))
model.add(layers.Dense(200, activation='tanh', name="Hidden_Layer_2")))
model.add(layers.Dense(100, activation='tanh', name="Hidden_Layer_3"))
# Capa output
model.add(layers.Dense(1, activation='linear'))
simulation=0
model.summary()    
for batch_size in [64,300,3000,30000]:   
    for LR in [1,1e-2,1e-3,1e-4,1e-5]:
        simulation+=1; print("simulation",simulation)
        model.compile(    
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
            loss='mse',
            metrics=['mse']
        )   
        history=model.fit(x_train,
            y_train,
            epochs=50,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1)   
        for value in history.history["mse"]:
            the_output+=[value]

with open("/home/carlos/red_neuronal_dipolos/miguel/batch_size_study/u2.txt", "w") as output:
            for value in the_output:
                output.write(str(value)+",")








