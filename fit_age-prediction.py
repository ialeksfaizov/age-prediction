#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd

from PIL import Image, ImageChops

from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Conv2D, AvgPool2D, Flatten, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

optimizer = Adam(lr=0.0001)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


# In[1]:


def load_train(path):
    df = pd.read_csv('/datasets/faces/labels.csv')
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       validation_split=0.2,
                                       horizontal_flip=True,
                                       rotation_range=20,
                                       zoom_range=[0.8,1.0],
                                       width_shift_range=0.15,
                                       height_shift_range=0.15,
                                       brightness_range=[0.2,1.0])
    train_gen_flow = train_datagen.flow_from_dataframe(dataframe=df,
                                                       directory=path + 'final_files/', 
                                                       x_col='file_name',
                                                       y_col='real_age',
                                                       target_size=(224, 224),
                                                       batch_size=16,
                                                       class_mode='raw',
                                                       subset='training',
                                                       seed=12345)
    return (train_gen_flow)


def load_test(path):
    df = pd.read_csv('/datasets/faces/labels.csv')
    valid_datagen = ImageDataGenerator(rescale=1./255,
                                       validation_split=0.2)
    validation_gen_flow = valid_datagen.flow_from_dataframe(dataframe=df,
                                                            directory=path + 'final_files/', 
                                                            x_col='file_name',
                                                            y_col='real_age',
                                                            target_size=(224, 224),
                                                            batch_size=16,
                                                            class_mode='raw',
                                                            subset='validation',
                                                            seed=12345)
    return (validation_gen_flow)

def create_model(input_shape):
    backbone = ResNet50(input_shape=input_shape,
                        weights='imagenet', 
                        include_top=False)
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=1, activation='relu'))
    model.compile(loss='mse', optimizer=optimizer, 
                  metrics=['mae'])
    return model

def train_model(model, train_datagen_flow, valid_datagen_flow, batch_size=None,
                steps_per_epoch=None, validation_steps=None,
                verbose=2, epochs=15):
    if steps_per_epoch is None:
      steps_per_epoch = len(train_datagen_flow)
    if validation_steps is None:
      validation_steps = len(valid_datagen_flow)
    model.fit(train_datagen_flow,
              validation_data = valid_datagen_flow,
              batch_size=batch_size,
              steps_per_epoch = steps_per_epoch,
              validation_steps = validation_steps,
              verbose=verbose, epochs=epochs)
    return model

