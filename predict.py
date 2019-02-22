#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

YAVUZ ÇETİN ÖLÜMSÜZDÜR

@author: Mehmet
"""

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.preprocessing import image


from PIL import Image
import io

import numpy as np

width = 197
height = 197
target = (width, height)

model = load_model('monkeys.h5')

def _predict(model, img_path):
    img = Image.open(img_path)
    img = img.resize(target)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    img = imagenet_utils.preprocess_input(img)
    print(img.shape)
    
    predictions = model.predict(img)
    results = imagenet_utils.decode_predictions([predictions])
    
    print(results)


def predict(model, img_path):
    img = image.load_img(img_path, target_size=(197, 197))
    x = image.img_to_array(img)
    print(x.shape)
    x = preprocess_input(x)
    s = x.shape
    x = np.reshape(x, (1, s[0], s[1], 3))
    preds = model.predict(x)

    # decode the results into a list of tuples (class, description, probability)
    print('Predicted:', np.argmax(preds, axis=1))  

predict(model, 'ersun-yanal.png')
    
