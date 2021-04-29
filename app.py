# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:24:48 2021

@author: amarj
"""

from __future__ import division, print_function
from flask import Flask,render_template,redirect, url_for,request
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
import sys
import os
import glob
import re
import numpy as np
from werkzeug.utils import secure_filename

app=Flask(__name__)
model_path="cotton_prd.h5"
model=load_model(model_path)

def mode(img_path,model):
    img=image.load_img(img_path,target_size=(224,244))
    x=image.img_to_array(img)
    x=x/225
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The leaf is diseased cotton leaf"
    elif preds==1:
        preds="The leaf is diseased cotton plant"
    elif preds==2:
        preds="The leaf is fresh cotton leaf"
    else:
        preds="The leaf is fresh cotton plant"
        
    
    
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
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
