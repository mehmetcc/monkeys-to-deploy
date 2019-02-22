#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

YAVUZ ÇETİN ÖLÜMSÜZDÜR

@author: Mehmet
"""

"""
monkey_classifier.py
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Model



class RESNet:
    
    def __init__(self):
        
        # some globals to keep things tidy
        self.height = 197
        self.width = 197
        self.batch_size = 64
        self.seed = 100
        
        self.num_classes = 10
        
        self.image_shape = (self.width, self.height, 3)
        
        self._augment()
        self.model = self._model()
        self.model.compile("sgd", "categorical_crossentropy", metrics=["accuracy"])
        
        self.model.fit_generator(self.train_generator,
                                 steps_per_epoch=2000 // self.batch_size,
                                 epochs=50,
                                 validation_data=self.validation_generator,
                                 validation_steps=800 // self.batch_size)

        
        self.model.save('monkeys.h5')

        
    
    def _augment(self):

        # augmentation of the dataset
        train_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        self.train_generator = train_datagen.flow_from_directory(
            'data/training',
            target_size=(self.height, self.width),
            batch_size=self.batch_size,
            seed=self.seed,
            shuffle=True,
            class_mode='categorical')

        # Test generator
        test_datagen = ImageDataGenerator()

        self.validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(self.height, self.width),
            batch_size=self.batch_size,
            seed=self.seed,
            shuffle=False,
            class_mode='categorical')

        self.train_num = self.train_generator.samples
        self.validation_num = self.validation_generator.samples 
        
    def _model(self):
        model = Sequential()
        base = ResNet50(include_top=False, pooling='avg', weights='imagenet')
        model.add(base)
        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.layers[0].trainable = False
        
        return model
        
        



res = RESNet()
