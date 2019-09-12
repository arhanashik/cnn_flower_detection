#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:26:34 2019

@author: MD. HASNAIN
"""

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Import Images
datagen = ImageDataGenerator()

train_it = datagen.flow_from_directory('oxfordflower17/train/', class_mode='binary', target_size = (32,32), batch_size = 32)
val_it = datagen.flow_from_directory('oxfordflower17/val/', class_mode='binary', target_size = (32,32), batch_size = 32)
test_it = datagen.flow_from_directory('oxfordflower17/test/', class_mode='binary', target_size = (32,32), batch_size = 32)

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#y_train = np_utils.to_categorical(y_train, 10)
#y_test = np_utils.to_categorical(y_test, 10)

batchX, batchY = train_it.next()
#print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
        train_it,
        steps_per_epoch=2000,
        epochs=30,
        validation_data=val_it,
        validation_steps=800)
#model.fit_generator(
#        datagen.flow(x_train, y_train, batch_size=60),
#        steps_per_epoch=2000,
#        epochs=30,
#        validation_data=(x_test, y_test),
#        validation_steps=800)
model.save_weights('first_try.h5')

