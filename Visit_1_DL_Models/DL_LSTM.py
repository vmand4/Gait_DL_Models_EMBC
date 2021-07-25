# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:03:31 2019

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
from keras.layers import Dense,Dropout
from keras.layers import LSTM
import keras.backend as K

# Model Creation
def CreateLSTM():
    LSTM_model = None # Clearing the NN.
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(32,input_shape=(9,1),return_sequences=True))
    LSTM_model.add(LSTM(32))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(Dense(256, activation='relu'))  
    LSTM_model.add(Dropout(0.5))
    LSTM_model.add(Dense(num_classes,activation = 'softmax'))
    LSTM_model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    LSTM_model.summary()
    return LSTM_model

# GPU memory clearning
def limit_memory(): 
    K.clear_session()
    time.sleep(2)
    gc.collect() 



# Walk AVG10 Training and Testing

batch_size = 32
epochs = 100
num_classes = 2
seed = 7
np.random.seed(seed)

#Visit 1 data and label seperation

Walk_AVG10_LSTM_Visit1 = pd.read_csv('Walk_AVG10_New_Rapid_Visit1.csv')
Walk_AVG10_LSTM_Visit1.set_index('ID')
Walk_AVG10_labels_visit1 = Walk_AVG10_LSTM_Visit1.iloc[:,10]
Walk_AVG10_labels_visit1 = Walk_AVG10_labels_visit1.replace("L", 1) 
Walk_AVG10_labels_visit1 = Walk_AVG10_labels_visit1.replace("R", 0)
Walk_AVG10_Data_Visit1 = Walk_AVG10_LSTM_Visit1.iloc[:,1:10]
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
    Model_5 = CreateLSTM()
    Model_5.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    
    # Model Training
    history = Model_5.fit(Walk_AVG10_Data_Visit1_train, Walk_AVG10_labels_visit1_train, batch_size=batch_size,epochs=epochs,verbose=1)
    
    # Trained model is used for prediction
    LSTM_predicted_classes=Model_5.predict(Walk_AVG10_Data_Visit1_test)
    LSTM_predicted_classes = np.argmax(np.round(LSTM_predicted_classes),axis=1)
    
    # Evaluating test data
    LSTM_Confusion = confusion_matrix(Walk_AVG10_labels_visit1_test, LSTM_predicted_classes)
    print(LSTM_Confusion)
    LSTM_Accuracy = accuracy_score(Walk_AVG10_labels_visit1_test, LSTM_predicted_classes)
    LSTM_AUC = roc_auc_score(Walk_AVG10_labels_visit1_test, LSTM_predicted_classes)
    LSTM_KAPPA = cohen_kappa_score(Walk_AVG10_labels_visit1_test, LSTM_predicted_classes)
    AUC_SCORES.append(LSTM_AUC)
    Accuracy.append(LSTM_Accuracy)
    KAPPA_SCORES.append((LSTM_KAPPA))
    
    # Delete Trained Model
    del history
    
    # Increment fold
    i = i+1
    
    # Clearning GPU memory
    limit_memory()

# Final Cross Validation Performance metrics Visit 1
Walk_AUC_AVG10_LSTM_Visit1 =  np.mean(AUC_SCORES)
Walk_Accuracy_AVG10_LSTM_Visit1 =  np.mean(Accuracy)
Walk_Kappa_AVG10_LSTM_Visit1 = np.mean(KAPPA_SCORES)
print("Walk AVG10 data visit 1 AUC, Accuracy, Kappa", Walk_Accuracy_AVG10_LSTM_Visit1, Walk_AUC_AVG10_LSTM_Visit1, Walk_Kappa_AVG10_LSTM_Visit1)


# Jogging Data Analysis


# Jog AVG10 Training and Testing

batch_size = 32
epochs = 100
num_classes = 2
seed = 7
np.random.seed(seed)

#Visit 1 data and label seperation

Jog_AVG10_LSTM_Visit1 = pd.read_csv('Jog_AVG10_New_Rapid_Visit1.csv')
Jog_AVG10_LSTM_Visit1.set_index('ID')
Jog_AVG10_labels_visit1 = Jog_AVG10_LSTM_Visit1.iloc[:,10]
Jog_AVG10_labels_visit1 = Jog_AVG10_labels_visit1.replace("L", 1) 
Jog_AVG10_labels_visit1 = Jog_AVG10_labels_visit1.replace("R", 0)
Jog_AVG10_Data_Visit1 = Jog_AVG10_LSTM_Visit1.iloc[:,1:10]
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
    Model_11 = CreateLSTM()
    Model_11.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    
    # Model Training
    history = Model_11.fit(Jog_AVG10_Data_Visit1_train, Jog_AVG10_labels_visit1_train, batch_size=batch_size,epochs=epochs,verbose=1)
    
    # Trained model is used for prediction
    LSTM_predicted_classes=Model_11.predict(Jog_AVG10_Data_Visit1_test)
    LSTM_predicted_classes = np.argmax(np.round(LSTM_predicted_classes),axis=1)
    
    # Evaluating test data
    LSTM_Confusion = confusion_matrix(Jog_AVG10_labels_visit1_test, LSTM_predicted_classes)
    print(LSTM_Confusion)
    LSTM_Accuracy = accuracy_score(Jog_AVG10_labels_visit1_test, LSTM_predicted_classes)
    LSTM_AUC = roc_auc_score(Jog_AVG10_labels_visit1_test, LSTM_predicted_classes)
    LSTM_KAPPA = cohen_kappa_score(Jog_AVG10_labels_visit1_test, LSTM_predicted_classes)
    AUC_SCORES.append(LSTM_AUC)
    Accuracy.append(LSTM_Accuracy)
    KAPPA_SCORES.append((LSTM_KAPPA))
    
    # Delete Trained Model
    del history
    
    # Increment fold
    i = i+1
    
    # Clearning GPU memory
    limit_memory()

# Final Cross Validation Performance metrics Visit 1
Jog_AUC_AVG10_LSTM_Visit1 =  np.mean(AUC_SCORES)
Jog_Accuracy_AVG10_LSTM_Visit1 =  np.mean(Accuracy)
Jog_Kappa_AVG10_LSTM_Visit1 = np.mean(KAPPA_SCORES)
print("Jog AVG10 data visit 1 Accuracy,AUC,  Kappa", Jog_Accuracy_AVG10_LSTM_Visit1, Jog_AUC_AVG10_LSTM_Visit1, Jog_Kappa_AVG10_LSTM_Visit1)



# Outputs of all models
print("Walk AVG10 data visit 1 Accuracy, AUC,  Kappa", Walk_Accuracy_AVG10_LSTM_Visit1, Walk_AUC_AVG10_LSTM_Visit1, Walk_Kappa_AVG10_LSTM_Visit1)
print("Jog AVG10 data visit 1 Accuracy,AUC,  Kappa", Jog_Accuracy_AVG10_LSTM_Visit1, Jog_AUC_AVG10_LSTM_Visit1, Jog_Kappa_AVG10_LSTM_Visit1)
