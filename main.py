# -*- coding: utf-8 -*-º
"""
Created on Tue Jul 19 18:15:21 2022

@author: migui
"""

#importing libraries#
import pandas as pd 
import numpy as np
from functions import *
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import tensorflow as tf    
from tensorflow import keras 
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
import random
####### INPUTS ######
X= df.iloc[:,0:117]
save_output_train=True
output_train_file="/home/carlos/red_neuronal_dipolos/miguel/activation_function_study/exp/exp_train.txt"
save_output_test=True
output_test_file="/home/carlos/red_neuronal_dipolos/miguel/activation_function_study/exp/exp_test.txt"

batch_size=3000
switch_regularization=False
regularization="l1" # l1 or l2
value_regularization=1e-5
LR=1e-5  
EPOCS=1000
switch_drop_out=False
drop_out=0.05
save_model=True
model_name="model_mse.h5"
data_path="/home/carlos/red_neuronal_dipolos/miguel/DEFINITIVE_MODEL2/rotation_10000_photos.txt"
activation_function=exp
loss_function=my_RMSE    #OPTIONS: my_RMSE , mse_and_norm_mse ,"MSE"
#######################

df = pd.read_csv(data_path, sep=",",engine="c")
the_output_lost=[];the_output_val_lost=[]
devuelvo=[]

X= df.iloc[:,0:117]
y=df.iloc[:,117:120]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

x_train=x_train.to_numpy()
y_train=y_train.to_numpy()

norm = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)

model = keras.Sequential()
if (regularization=="l2"):
    kerner_regularizer_l1 = regularizers.l2(value_regularization)

if (regularization=="l1"):
    kerner_regularizer_l1 = regularizers.l1(value_regularization)


if(switch_drop_out==True):
    model.add(Dropout(drop_out))
#intermediate layers
model.add(layers.Dense(300, input_shape=(117,),activation=activation_function, name="Hidden_Layer_1"))
model.add(layers.Dense(200, activation=activation_function, name="Hidden_Layer_2"))
model.add(layers.Dense(100, activation=activation_function, name="Hidden_Layer_3"))
# output layer
model.add(layers.Dense(3, activation='linear'))
simulation=0
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=loss_function,
    metrics=['mse'])   
history=model.fit(x_train,
        y_train,
        epochs=EPOCS,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1)   

###### saving the process results #######
for value in history.history["val_mse"]:
    the_output_val_lost+=[value]
for value in history.history["mse"]:
    the_output_lost+=[value]
if(save_output_train==True):
    with open(output_train_file, "w") as output:
                for value in the_output_lost:
                    output.write(str(value)+",")


if(save_output_test==True):
    with open(output_test_file, "w") as output2:
                for value in the_output_val_lost:
                    output2.write(str(value)+",")
model.save(model_name)
