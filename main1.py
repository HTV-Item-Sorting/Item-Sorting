import os
import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import camera

from flask import Flask, render_template, Response

app = Flask(__name__, instance_relative_config=True)


@app.route('/hello')
def test(test_config=None):
    return 'Hello, World!'


@app.route('/video_feed')
def video_feed():
    return Response(camera.view(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def login():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)