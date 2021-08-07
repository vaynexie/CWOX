# Prapare dataset for HLTM training
# Input: Model, Training Set
# Output: Dataset used for HLTM training

# Load needed packages
import os
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io
from sklearn.metrics import accuracy_score
import gc
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
from tensorflow.keras.applications.resnet import (
    ResNet50,
    preprocess_input,
    decode_predictions,
)
net = ResNet50(weights='imagenet')

# Make predictions and save the probabilities
size =[224,224]
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = data_generator.flow_from_directory(directory='training_set/',
target_size=size,
batch_size=32,
shuffle=False)         
prediction_result = net.predict_generator(
    validation_generator, len(validation_generator), verbose=1,
    max_queue_size=10,workers=8
)
np.save(
     'prediction_result.npy',
    prediction_result,
)    
prediction_result=[]
validation_generator.reset()
tf.keras.backend.clear_session()


# Prepare data for HLTA training
data_list=np.load('prediction_result.npy',allow_pickle=True)

def numberlist(nums, limit):
    prefix = []
    sum = 0
    for num in nums:
        sum += num
        prefix.append(num)
        if sum > limit:
            return len(prefix)

# list_95 is the ouptut dataset used for HLTM training
list_95=[]
kk=0
for i in data_list:
    if kk%100==0:print(kk)
    temp=i
    sort_proba=np.sort(temp)[::-1]
    sort_index=np.argsort(temp)[::-1]
    cut_index=numberlist(sort_proba, 0.95)
    if len(sort_index[:cut_index])>1:
        list_95.append(sort_index[:cut_index])
    kk+=1
np.save('list_95.npy',list_95)

