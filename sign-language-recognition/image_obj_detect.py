#!flask/bin/python
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import base64
import json
# from datetime import datetime

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from collections import defaultdict
from utils import label_map_util
from protos import string_int_label_map_pb2
import multiprocessing
from multiprocessing import Queue, Pool
import time
import datetime
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
from keras.models import load_model
import sklearn
import skimage
from skimage.transform import resize
from keras.backend import clear_session

app = Flask(__name__)
CORS(app, support_credentials=True)

_score_thresh = 0.27




MODEL_NAME = 'hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

# In[5]:


NUM_CLASSES = 1
# load label map
label_map = label_map_util.load_labelmap('hand_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_inference_graph():
    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


video_source = 0
#        help='Device index of the camera.
num_hands=2
#        help='Max number of hands to detect.')
fps=1
#        help='Show FPS on detection/display visualization')
width=300
#        help='Width of the frames in the video stream.')
height=200
#        help='Height of the frames in the video stream.')
display=1
#        help='Display the detected images using OpenCV. This reduces FPS')
num_workers=4
#        help='Number of workers.')
queue_size=5
score_thresh = 0.2

print(">> loading frozen model for worker")


print('Loading model...')
global model_classification
model_classification = load_model('vgg16model.h5')
detection_graph, sess = load_inference_graph()
# global graph2
graph = tf.get_default_graph()
print('model loaded')

@app.route('/upload', methods=['POST'])
@cross_origin(supports_credentials=True)
def upload_base64_img():
    global graph
    with graph.as_default():
        def detect_objects(image_np, detection_graph, sess):
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores,
                    detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            return np.squeeze(boxes), np.squeeze(scores)

        def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
            for i in range(num_hands_detect):
               if (scores[i] > score_thresh):
                   (left, right, top, bottom) = (boxes[i][1] * im_width*0.8, boxes[i][3] * im_width*1.1,
                                                 boxes[i][0] * im_height*0.8, boxes[i][2] * im_height*1.1)
                   p1 = (int(left), int(top))
                   p2 = (int(right), int(bottom))
                   x= int(left)
                   y= int(top)
                   w=int(right)
                   h=int(bottom)
                   roi=image_np[y:y+h,x:x+w]
                   #cv2.imwrite('image.png',roi)
                   map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: '-', 28: 'space'}
                   X_ = []
                   imageSize=50
                   img_file_=roi
                   img_file_ = skimage.transform.resize(img_file_, (imageSize, imageSize, 3))
                   img_arr_ = np.asarray(img_file_)
                   X_.append(img_arr_)
                   X_ = np.asarray(X_)
                   y_pred = model_classification.predict(X_)
                   Y_pred_classes = np.argmax(y_pred,axis = 1)
                   print('Predicting the gesture')
                   print(map_characters.get(Y_pred_classes[0]))
                   cv2.imwrite('image.png', roi)
                   cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)

        def get_pred(image_path):
            # detection_graph, sess = load_inference_graph()
            sess = tf.Session(graph=detection_graph)
            #print("> ===== in worker loop, frame ", frame_processed)
            frame = cv2.imread(image_path)
            pred = '-'
            ACCURACY = 0
            #frame_processed = 0

            #import matplotlib.pyplot as plt
            #lt.imshow(frame)
            
            if (frame is not None):
                frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cap_params = {}
                cap_params['im_height'], cap_params['im_width'] ,x = frame.shape
                cap_params['score_thresh'] = score_thresh
                # max number of hands we want to detect/track
                cap_params['num_hands_detect'] = num_hands
                # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
                # while scores contains the confidence for each of these boxes.
                # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

                boxes, scores = detect_objects(
                    frame, detection_graph, sess)
                # draw bounding boxes

                im_width = cap_params['im_width']
                im_height = cap_params['im_height']
                image_np = frame
                for i in range(cap_params['num_hands_detect']):
                    if (scores[i] > cap_params["score_thresh"]):
                        (left, right, top, bottom) = (boxes[i][1] * im_width*0.8, boxes[i][3] * im_width*1.1,
                                                    boxes[i][0] * im_height*0.8, boxes[i][2] * im_height*1.1)
                        # (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                        #                             boxes[i][0] * im_height, boxes[i][2] * im_height)
                        p1 = (int(left), int(top))
                        p2 = (int(right), int(bottom))
                        x= int(left)
                        y= int(top)
                        w=int(right)
                        h=int(bottom)
                        roi=image_np[y:y+h,x:x+w]
                        #cv2.imwrite('image.png',roi)
                        map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}
                        X_ = []
                        imageSize=50
                        img_file_=roi
                        img_file_ = skimage.transform.resize(img_file_, (imageSize, imageSize, 3))
                        img_arr_ = np.asarray(img_file_)
                        X_.append(img_arr_)
                        X_ = np.asarray(X_)
                        y_pred = model_classification.predict(X_)
                        Y_pred_classes = np.argmax(y_pred,axis = 1)
                        ACCURACY=y_pred[0][Y_pred_classes[0]]*100
                        # if (ACCURACY > 80):
                        print('Predicting the gesture')
                        print(map_characters.get(Y_pred_classes[0]))
                        pred = map_characters.get(Y_pred_classes[0])
                        cv2.imwrite('roi.jpg', roi)
                        # cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
                # add frame annotated with bounding box to queue
                #cv2.imwrite('image_obj_det_hands.png',frame)
                #frame_processed += 1
            # clear_session()
            # sess.close()
            return pred, ACCURACY

        content = request.get_json()
        # checking if the image is present or not.
        if 'image' not in content:
            # abort(400)
            # abort(Response('No Image data received'))
            return 'Image not received'

        imgdata = base64.b64decode(content['image'])

        # (dt, micro) = datetime.utcnow().strftime('%Y%m%d%H%M%S.%f').split('.')
        # dt = "%s%03d" % (dt, int(micro) / 1000)
        filename = 'webcamImg/some_image.jpg'
        with open(filename, 'wb') as f:
            f.write(imgdata)
        # print("--------------------->>>>>>", filename)
        letter, acc = get_pred(filename)

        result = {
            "hand_object": "http://localhost:8888/ada/roi.jpg",
            "letter_detected": str(letter),
            "accuracy": acc
        }

        return json.dumps(result)

app.run(port=5000, debug=True)