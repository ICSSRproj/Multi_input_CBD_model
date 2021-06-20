#!/usr/bin/env python
# coding: utf-8

# In[119]:


import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[120]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model


# In[163]:


IMG_SIZE = 90
list_classes = ["Toxic", "Severe_toxic", "Obscene", "Threat", "Insult", "Identity Hate"]


# In[164]:


from flask import Flask,request,jsonify
from flask_cors import CORS
import json


# In[168]:


#load model
model = load_model("mm01_increased accuracy_19_06_2021_2.h5")


# In[169]:


app = Flask(__name__)
CORS(app)


# In[170]:

@app.route('/')
def man():
    return "CYBER BULLYING DETECTION BY DEEP LEARNING MULTI INPUT"
@app.route('/check', endpoint='check', methods=['POST'])
def check():
    #here the "user" is used i don't even know why
    dat = request.get_json("user")
#     print(dat)
    text=dat["TXT"]
    img_base64=dat["BASE64"]
    print(text)
    print(base64)
    try:
        
        #txt processing
        txt=[None]
        txt[0]=text
        max_features = 20000
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(txt))
        list_tokenized_txt = tokenizer.texts_to_sequences(txt)
        maxlen = 200
        X_t = pad_sequences(list_tokenized_txt, maxlen=maxlen)

        #image processing
        im_bytes = base64.b64decode(img_base64)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv2.imdecode(im_arr, flags=cv2.COLOR_BGR2RGB)
        new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        n_img = np.expand_dims(new_array, axis=0)
        n_img=n_img/255

        #model prediction
        pred = model.predict([n_img,X_t])
        image_pred=pred[0]
        txt_pred=pred[1]
        img_res=np.argmax(image_pred, axis=-1)
        if(img_res[0]==0):
            img_res2="Cyber bullying Not Detected"
        else:
            img_res2="Cyber bullying Detected"
        print("Image Prediction : ",img_res2)

        prediction_labels = []

        for i,label in enumerate(list_classes):
            label_prob = txt_pred[:,i]

            if label_prob>0.7:
                prediction_labels.append(label)
        print("Text Prediction : ",prediction_labels)
        arr=[]
        arr.append({"IMG_RES":img_res2})
        for label in prediction_labels:
            arr.append({"TXT_RES":label})
    #     res={"IMG_RES" : img_res, "TXT_RES" : prediction_labels}
        print(arr)
    #     return jsonify("test")
        return jsonify(arr)
#     try:
#         return jsonify(res)
    except:
        return "0"


# In[171]:


if __name__ == '__main__':
   app.run()
