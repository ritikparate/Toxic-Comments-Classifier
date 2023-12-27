#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
from flask import request
from flask import Response
from markupsafe import escape
from flask import Flask, render_template, session
import pickle

from distutils.log import debug
from fileinput import filename
import pandas as pd
from werkzeug.utils import secure_filename

from tensorflow.keras.layers import TextVectorization
import tensorflow
from txt_analysis import creat_cloud, word_distribution

os.chdir(r"C:\Users\Renu\Downloads\P316")
UPLOAD_FOLDER = os.path.join('static_files', 'uploads')

# Define allowed files
ALLOWED_EXTENSIONS = {'csv','txt','png'}

app = Flask(__name__)

# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'This is your secret key to utilize session in Flask'

@app.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
 # upload file flask
        f = request.files.get('file')
        f.filename = "upload.txt"
        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)

        f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                            data_filename))

        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],data_filename)

        return render_template('uploaded.html')
    return render_template("classification.html")

@app.route('/show_data')
def showData():
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read text
    with open(data_file_path, "r") as file:
        content = file.read()
        print(content)
    return render_template('classification.html',
                        data_var=content)


@app.route('/results')
def result(): 
    #unpickle the vectorizer and model
    vect_pickled = pickle.load(open(r"C:\Users\Renu\Downloads\P316\models\vect.pkl", "rb"))
    model = pickle.load(open('models\model.pkl','rb'))
    vectorizer = TextVectorization.from_config(vect_pickled['config'])
    vectorizer.set_weights(vect_pickled['weights'])
    
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read text
    with open(data_file_path, "r") as file:
        content = file.read()
    creat_cloud(content)
    prediction = word_distribution(content)
    return render_template('result.html', prediction_text='Toxicity analysis: {}'.format(prediction))
    return render_template('loading.html')
    

if __name__ == '__main__':
    app.run(debug=True)


