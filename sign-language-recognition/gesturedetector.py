import skimage
from skimage.transform import resize
from keras.models import load_model
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
print('Loading model...')
model = load_model('vgg16model.h5') 
print('model loaded')


map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}
while (True):
    X_ = []
    imageSize=50
    img_file_ = cv2.imread('image.png')
    if img_file_ is not None:
        img_file_ = skimage.transform.resize(img_file_, (imageSize, imageSize, 3))
        img_arr_ = np.asarray(img_file_)
        X_.append(img_arr_)
        X_ = np.asarray(X_)
        y_pred = model.predict(X_)
        Y_pred_classes = np.argmax(y_pred,axis = 1)
        print('Predicting the gesture')
        print(map_characters.get(Y_pred_classes[0]))

