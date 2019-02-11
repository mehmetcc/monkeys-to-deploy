from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


width = 197
height = 197

result_dictionary = {0: 'Mantled Howler (Alouatta Palliata)',
                     1: 'Patas Monkey (Erythrocebus Patas)',
                     2: 'Bald Uakari (Cacajao Calvus)',
                     3: 'Japanese Macaque (Macaca Fuscata)',
                     4: 'Pygmy Marmoset (Cebuella Pygmea)',
                     5: 'White Headed Capuchin (Cebus Capucinus)',
                     6: 'Silvery Marmoset (Mico Argentatus)',
                     7: 'Common Squirrel Monkey (Saimiri Sciureus)',
                     8: 'Black Headed Night Monkey (Aotus Nigriceps)',
                     9: 'Nilgiri Langur (Trachypithecus Johnii)'}

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/monkeys.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(197, 197))
    x = image.img_to_array(img)
    print(x.shape)
    x = preprocess_input(x)
    s = x.shape
    x = np.reshape(x, (1, s[0], s[1], 3))
    preds = np.argmax(model.predict(x), axis=1)[0]

    # decode the results into a list of tuples (class, description, probability)
    print('Predicted:', preds)

    return preds



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Clean up
        os.remove(file_path)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        result = 'This guy looks like a ' + result_dictionary[preds]

        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
