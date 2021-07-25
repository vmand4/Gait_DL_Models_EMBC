# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:08:06 2019

@author: varunm1
"""


import numpy as np, time, gc
from keras.utils import to_categorical
import pandas as pd
from sklearn.metrics import confusion_matrix
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, cohen_kappa_score,accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers import  GRU
import keras.backend as K

# Model Creation
def CreateGRU():
    GRU_model = None # Clearing the NN.
    GRU_model = Sequential()
    GRU_model.add(GRU(32,input_shape=(9,1),return_sequences=True))
    GRU_model.add(GRU(32))
    GRU_model.add(Dropout(0.2))
    GRU_model.add(Dense(256, activation='relu'))  
    GRU_model.add(Dropout(0.5))
    GRU_model.add(Dense(num_classes,activation = 'softmax'))
    GRU_model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    GRU_model.summary()
    return GRU_model

# GPU memory clearning
def limit_memory(): 
    K.clear_session()
    time.sleep(10)
    gc.collect() 



# Walk AVG10 Training and Testing

batch_size = 32
epochs = 100
num_classes = 2
seed = 7
np.random.seed(seed)

#Visit 1 data and label seperation

Walk_AVG10_GRU_Visit1 = pd.read_csv('Walk_AVG10_New_Rapid_Visit1.csv')
Walk_AVG10_GRU_Visit1.set_index('ID')
Walk_AVG10_labels_visit1 = Walk_AVG10_GRU_Visit1.iloc[:,10]
Walk_AVG10_labels_visit1 = Walk_AVG10_labels_visit1.replace("L", 1) 
Walk_AVG10_labels_visit1 = Walk_AVG10_labels_visit1.replace("R", 0)
Walk_AVG10_Data_Visit1 = Walk_AVG10_GRU_Visit1.iloc[:,1:10]
Walk_AVG10_Data_Visit1= np.array(Walk_AVG10_Data_Visit1)
Walk_AVG10_labels_visit1 = np.array(Walk_AVG10_labels_visit1)
Walk_AVG10_Data_Visit1 = np.expand_dims(Walk_AVG10_Data_Visit1, axis=2)

# Initializing Cross Validation Parameters
kf = KFold(n_splits=5, shuffle=True)
kf.get_n_splits(Walk_AVG10_Data_Visit1)

# Initializingt Performance Metrics
AUC_SCORES = []
KAPPA_SCORES = []
Accuracy = []

# Initializing Fold Count
i = 1

# Cross Validation on Visit 1 Data
for train, test in kf.split(Walk_AVG10_Data_Visit1):
    
    # Fold Count
    print("Running Fold", i)
    
    # Data and labels extraction for Training and testing in current fold of Cross Validation
    Walk_AVG10_Data_Visit1_train = Walk_AVG10_Data_Visit1[train]
    Walk_AVG10_labels_visit1_train = Walk_AVG10_labels_visit1[train]
    Walk_AVG10_labels_visit1_train = to_categorical(Walk_AVG10_labels_visit1_train)
    Walk_AVG10_Data_Visit1_test = Walk_AVG10_Data_Visit1[test]
    Walk_AVG10_labels_visit1_test = Walk_AVG10_labels_visit1[test]
    Walk_AVG10_labels_visit1_test_Shot = to_categorical(Walk_AVG10_labels_visit1_test)
    
    # Model initialization 
    Model_5 = CreateGRU()
    Model_5.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    
    # Model Training
    history = Model_5.fit(Walk_AVG10_Data_Visit1_train, Walk_AVG10_labels_visit1_train, batch_size=batch_size,epochs=epochs,verbose=1)
    
    # Trained model is used for prediction
    GRU_predicted_classes=Model_5.predict(Walk_AVG10_Data_Visit1_test)
    GRU_predicted_classes = np.argmax(np.round(GRU_predicted_classes),axis=1)
    
    # Evaluating test data
    GRU_Confusion = confusion_matrix(Walk_AVG10_labels_visit1_test, GRU_predicted_classes)
    print(GRU_Confusion)
    GRU_Accuracy = accuracy_score(Walk_AVG10_labels_visit1_test, GRU_predicted_classes)
    GRU_AUC = roc_auc_score(Walk_AVG10_labels_visit1_test, GRU_predicted_classes)
    GRU_KAPPA = cohen_kappa_score(Walk_AVG10_labels_visit1_test, GRU_predicted_classes)
    AUC_SCORES.append(GRU_AUC)
    Accuracy.append(GRU_Accuracy)
    KAPPA_SCORES.append((GRU_KAPPA))
    
    # Delete Trained Model
    del history
    
    # Increment fold
    i = i+1
    
    # Clearning GPU memory
    limit_memory()

# Final Cross Validation Performance metrics Visit 1
Walk_AUC_AVG10_GRU_Visit1 =  np.mean(AUC_SCORES)
Walk_Accuracy_AVG10_GRU_Visit1 =  np.mean(Accuracy)
Walk_Kappa_AVG10_GRU_Visit1 = np.mean(KAPPA_SCORES)
print("Walk AVG10 data visit 1 AUC, Accuracy, Kappa", Walk_Accuracy_AVG10_GRU_Visit1, Walk_AUC_AVG10_GRU_Visit1, Walk_Kappa_AVG10_GRU_Visit1)


# Jogging Data Analysis


# Jog AVG10 Training and Testing

batch_size = 32
epochs = 100
num_classes = 2
seed = 7
np.random.seed(seed)

#Visit 1 data and label seperation

Jog_AVG10_GRU_Visit1 = pd.read_csv('Jog_AVG10_New_Rapid_Visit1.csv')
Jog_AVG10_GRU_Visit1.set_index('ID')
Jog_AVG10_labels_visit1 = Jog_AVG10_GRU_Visit1.iloc[:,10]
Jog_AVG10_labels_visit1 = Jog_AVG10_labels_visit1.replace("L", 1) 
Jog_AVG10_labels_visit1 = Jog_AVG10_labels_visit1.replace("R", 0)
Jog_AVG10_Data_Visit1 = Jog_AVG10_GRU_Visit1.iloc[:,1:10]
Jog_AVG10_Data_Visit1= np.array(Jog_AVG10_Data_Visit1)
Jog_AVG10_labels_visit1 = np.array(Jog_AVG10_labels_visit1)
Jog_AVG10_Data_Visit1 = np.expand_dims(Jog_AVG10_Data_Visit1, axis=2)

# Initializing Cross Validation Parameters
kf = KFold(n_splits=5, shuffle=True)
kf.get_n_splits(Jog_AVG10_Data_Visit1)

# Initializingt Performance Metrics
AUC_SCORES = []
KAPPA_SCORES = []
Accuracy = []

# Initializing Fold Count
i = 1

# Cross Validation on Visit 1 Data
for train, test in kf.split(Jog_AVG10_Data_Visit1):
    
    # Fold Count
    print("Running Fold", i)
    
    # Data and labels extraction for Training and testing in current fold of Cross Validation
    Jog_AVG10_Data_Visit1_train = Jog_AVG10_Data_Visit1[train]
    Jog_AVG10_labels_visit1_train = Jog_AVG10_labels_visit1[train]
    Jog_AVG10_labels_visit1_train = to_categorical(Jog_AVG10_labels_visit1_train)
    Jog_AVG10_Data_Visit1_test = Jog_AVG10_Data_Visit1[test]
    Jog_AVG10_labels_visit1_test = Jog_AVG10_labels_visit1[test]
    Jog_AVG10_labels_visit1_test_Shot = to_categorical(Jog_AVG10_labels_visit1_test)
    
    # Model initialization 
    Model_11 = CreateGRU()
    Model_11.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    
    # Model Training
    history = Model_11.fit(Jog_AVG10_Data_Visit1_train, Jog_AVG10_labels_visit1_train, batch_size=batch_size,epochs=epochs,verbose=1)
    
    # Trained model is used for prediction
    GRU_predicted_classes=Model_11.predict(Jog_AVG10_Data_Visit1_test)
    GRU_predicted_classes = np.argmax(np.round(GRU_predicted_classes),axis=1)
    
    # Evaluating test data
    GRU_Confusion = confusion_matrix(Jog_AVG10_labels_visit1_test, GRU_predicted_classes)
    print(GRU_Confusion)
    GRU_Accuracy = accuracy_score(Jog_AVG10_labels_visit1_test, GRU_predicted_classes)
    GRU_AUC = roc_auc_score(Jog_AVG10_labels_visit1_test, GRU_predicted_classes)
    GRU_KAPPA = cohen_kappa_score(Jog_AVG10_labels_visit1_test, GRU_predicted_classes)
    AUC_SCORES.append(GRU_AUC)
    Accuracy.append(GRU_Accuracy)
    KAPPA_SCORES.append((GRU_KAPPA))
    
    # Delete Trained Model
    del history
    
    # Increment fold
    i = i+1
    
    # Clearning GPU memory
    limit_memory()

# Final Cross Validation Performance metrics Visit 1
Jog_AUC_AVG10_GRU_Visit1 =  np.mean(AUC_SCORES)
Jog_Accuracy_AVG10_GRU_Visit1 =  np.mean(Accuracy)
Jog_Kappa_AVG10_GRU_Visit1 = np.mean(KAPPA_SCORES)
print("Jog AVG10 data visit 1 Accuracy,AUC,  Kappa", Jog_Accuracy_AVG10_GRU_Visit1, Jog_AUC_AVG10_GRU_Visit1, Jog_Kappa_AVG10_GRU_Visit1)

# Output of All Models

Print("Walk AVG10 data visit 1 Accuracy, AUC,  Kappa", Walk_Accuracy_AVG10_GRU_Visit1, Walk_AUC_AVG10_GRU_Visit1, Walk_Kappa_AVG10_GRU_Visit1)
print("Jog AVG10 data visit 1 Accuracy,AUC,  Kappa", Jog_Accuracy_AVG10_GRU_Visit1, Jog_AUC_AVG10_GRU_Visit1, Jog_Kappa_AVG10_GRU_Visit1)
print("Walk_Jog raw data visit 1 Accuracy, AUC,  Kappa", Walk_Jog_Accuracy_Raw_GRU_Visit1, 