# coding: utf-8

# Object Detection Demo

# Imports
import developed as dev
dev.print_stamp()

from datetime import datetime
import numpy as np
import boto3
import smtplib 
from tkinter import messagebox
from pygame import mixer
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
import matplotlib.pyplot as plt



# Env setup
# This is needed to display the images.
# get_ipython().magic(u'matplotlib inline')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
client = boto3.client(
    "sns",
    aws_access_key_id="XXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    aws_secret_access_key="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    region_name="eu-west-1"
)


# Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util

from utils import visualization_utils as vis_util
#mail
# Model preparation

# Variables
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# Uncomment the following lines to download the model (required)
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# Download Model
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())

# Load a (frozen) Tensorflow model into memory.


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `drone`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        video_path = '/media/suresh_arunachalam/user/Project/python_coding/tensorflow_video_object_detection-master1/object_detection/hello4.mp4'
        cap = cv2.VideoCapture(video_path)
        ret = True
    
        value=[0]
        value1=[0]
        value2=[0]
        value3=[0]
                
        drone=0
        bird=0
        kite=0
        person=0
        def cal():
                #os.system('spd-say "Alert"')
                #messagebox.showwarning("Warning","Warning message")
                 mixer.init()
                 mixer.music.load("mysong.wav")
                 mixer.music.play()
                # sending the mail
                 """ 
                 a8=str(datetime.now())
                 print(a8)
                 s = smtplib.SMTP('smtp.gmail.com', 587)


# start TLS for security 
                 s.starttls() 

# Authentication 
                 s.login("from@gmail.com","from_password") 

                 message = ("Drone is entered into our Area  !!! By Camera - 6")

                 
                 client.publish(PhoneNumber="+918778479731", Message="Drone is entered into our Area!!! By, Camera-6")
                 s.sendmail("from@gmail.com", "to@gmail.com", message) 
                 print("Mail & SMS Alert Sent")
# terminating the session 
                 s.quit() 

                  """
        while ret:
            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            print(num)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            print(classes)
            print(scores[0][0]*100)
        
            if classes[0][0]==5:
                drone=1
                value.append((scores[0][0])*100)
                if scores[0][0]*100>20:
                    print("Drone")
                    print(scores[0][0]*100)
                    cal()
                
            elif classes[0][0]==1:
                person=1
                value1.append((scores[0][0])*100)
                                
            elif classes[0][0]==16:
                bird=1
                value2.append((scores[0][0])*100)
                                
            elif classes[0][0]==38:
                kite=1
                value3.append((scores[0][0])*100)
            if drone==1:
                d1="Drone"
            if person==1:
                p1="Person"
            if bird==1:
                b1="Bird"
            if kite==1:
                k1="Kite"
            
            cv2.imshow('Video Detection', cv2.resize(image_np, (900, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                if drone==1 and person==1 and bird==1 and kite==1:
                    plt.plot(value,'g-')
                    plt.plot(value1,'b-')
                    plt.plot(value2,'y-')
                    plt.plot(value3,'r-')
                
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Green = '+d1,'Blue = '+p1,'Yellow = '+b1,'Red = '+k1],loc='lower right')
                    plt.show()
                if drone==0 and person==0 and bird==1 and kite==1:

                    plt.plot(value2,'y-')
                    plt.plot(value3,'r-')
                
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Yellow = '+b1,'Red = '+k1],loc='lower right')
                    plt.show()
                if drone==0 and person==0 and bird==0 and kite==1:

                    plt.plot(value3,'r-')
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Red = '+k1],loc='lower right')
                    plt.show()

                if drone==0 and person==0 and bird==1 and kite==0:

                    plt.plot(value2,'y-')

                
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Yellow = '+b1],loc='lower right')
                    plt.show()

                if drone==0 and person==1 and bird==0 and kite==0:

                    plt.plot(value1,'b-')

                
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Blue = '+p1],loc='lower right')
                    plt.show()

                
                if drone==0 and person==1 and bird==0 and kite==1:
                    plt.plot(value1,'b-')

                    plt.plot(value3,'r-')

                
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Blue = '+p1,'Red = '+k1],loc='lower right')
                    plt.show()                
                
                if drone==0 and person==1 and bird==1 and kite==0:

                    plt.plot(value1,'b-')
                    plt.plot(value2,'y-')

                
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Blue = '+p1,'Yellow = '+b1],loc='lower right')
                    plt.show()                
                if drone==0 and person==1 and bird==1 and kite==1:

                    plt.plot(value1,'b-')
                    plt.plot(value2,'y-')
                    plt.plot(value3,'r-')
                
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Blue = '+p1,'Yellow = '+b1,'Red = '+k1],loc='lower right')
                    plt.show()

                if drone==1 and person==0 and bird==0 and kite==0:
                    plt.plot(value,'g-')

                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Green = '+d1],loc='lower right')
                    plt.show()
                if drone==1 and person==0 and bird==0 and kite==1:
                    plt.plot(value,'g-')

                    plt.plot(value3,'r-')
                
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Green = '+d1,'Red = '+k1],loc='lower right')
                    plt.show()

                if drone==1 and person==0 and bird==1 and kite==0:
                    plt.plot(value,'g-')

                    plt.plot(value2,'y-')

                
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Green = '+d1,'Yellow = '+b1],loc='lower right')
                    plt.show()

                if drone==1 and person==0 and bird==1 and kite==1:
                    plt.plot(value,'g-')
                    plt.plot(value2,'y-')
                    plt.plot(value3,'r-')
                
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Green = '+d1,'Yellow = '+b1,'Red = '+k1],loc='lower right')
                    plt.show()
                if drone==1 and person==1 and bird==0 and kite==0:
                    plt.plot(value,'g-')
                    plt.plot(value1,'b-')

                
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Green = '+d1,'Blue = '+p1],loc='lower right')
                    plt.show()
                if drone==1 and person==1 and bird==0 and kite==1:
                    plt.plot(value,'g-')
                    plt.plot(value1,'b-')
                    plt.plot(value3,'r-')
                
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Green = '+d1,'Blue = '+p1,'Red = '+k1],loc='lower right')
                    plt.show()
                if drone==1 and person==1 and bird==1 and kite==0:
                    plt.plot(value,'g-')
                    plt.plot(value1,'b-')
                    plt.plot(value2,'y-')

                
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.legend(['Green = '+d1,'Blue = '+p1,'Yellow = '+b1],loc='lower right')
                    plt.show()

                if drone==0 and person==0 and bird==0 and kite==0:
                    plt.ylabel('Confidence(%)')
                    plt.xlabel('Time(sec)')
                    plt.show()

                                
                cv2.destroyAllWindows()
                cap.release()    
                break
            
