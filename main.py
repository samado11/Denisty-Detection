#!/usr/bin/env python
# coding: utf-8

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from matplotlib import cm as CM

# from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
# from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

import sys

# In[9]:
#%%
import math
import cv2
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model import SASNet
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from matplotlib import cm as CM
import numpy as np
from datetime import datetime
from flask import Flask, flash, request, redirect, url_for
from flask_restful import Api, Resource, reqparse
import requests
import os
from werkzeug.utils import secure_filename
from flask import jsonify
from flask import render_template
def save_density_map(density_map, name):
        plt.figure(dpi=250)
        plt.axis('off')
        plt.margins(0, 0)
        plt.imshow(density_map, cmap=CM.jet)
        plt.savefig(name, dpi=250, bbox_inches='tight', pad_inches=0)
        
def read_video(video_input):
    
    trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = SASNet(pretrained=False).cpu()
    model.load_state_dict(torch.load('models/SHHA.pth',map_location=torch.device('cpu')))
    lst = []
    videoFile = video_input
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            img= cv2.resize(frame , (250,250),interpolation = cv2.INTER_CUBIC)
            img = trans(img)
            img= torch.Tensor(img[None, :, :, :])
            img=model(img.cpu())
            img=np.squeeze(img.detach().numpy(), axis=1)
            img=np.squeeze(img, axis=0)
            save_density_map(img,str(x))
            #filename =   str(int(x)) + '.jpg'
            #cv2.imwrite(filename, img)
            x+=1

    cap.release()

    img_array = []
    for filename in glob.glob('*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter('static/project.mp4',cv2.VideoWriter_fourcc(*'MPV4'), 1, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return render_template('video.html')



app = Flask(__name__)
api = Api(app)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file :
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return read_video('uploads/'+file.filename), 200
# Adding routes to the Application and Endpoints to App.
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


