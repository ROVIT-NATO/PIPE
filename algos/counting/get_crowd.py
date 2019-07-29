
import time
import paho.mqtt.client as mqtt
import math
import datetime
import os
import numpy as np
import base64
import cv2
import json
import arrow
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parents[4]))

from algos.counting.C_CNN.src.crowd_count import CrowdCounter
import algos.counting.C_CNN.src.network as nw



scale = 0.3

model_path1 = '/home/mahdi/PycharmProjects/PIPE/algos/counting/C_CNN/final_models/new.h5'
net1 = CrowdCounter()
nw.load_net(model_path1, net1)

if net1.cuda_available():
    print('GPU Detected!')
    net1.cuda()
else:
    print('RUNNING WITHOUT CUDA SUPPORT')
    net1.eval()



def process_frame(frame):


        height = frame.shape[0]
        width = frame.shape[1]

        x = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

        # INFERENCE
        x = x.astype(np.float32, copy=False)
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)




        density_map = net1(x)
        density_map = density_map.data.cpu().numpy()[0][0]




        count = np.sum(density_map)

        # CONVERT BACK TO ORIGINAL SCALE
        density_map = cv2.resize(density_map, (width, height))




        # CREATE THE MESSAGE
        cam_id = 'camera_id'
        cam_bearing = 'camera_bearing'
        cam_pos = 'camera_position'
        timestamp = str(datetime.datetime.now())




        message = create_obs_message(count, density_map, timestamp, cam_id, cam_pos)


        print('Crowd count = ' + str(math.ceil(count)))

        return math.ceil(count), density_map



#
# # mqtt connection function
#
# def on_connect(client, userdata, flags, rc):
#     print("Connected with Code :" + str(rc))
#     # Subscribe Topic from here
#     client.subscribe("Test/#")
#
# # mqtt meg function
#
# def on_message(client, userdata, msg):
#     # print the message received from the subscribed topic
#     print(str(msg.payload))


def create_obs_message(count, density_map, timestamp, cam_id, cam_pos ):
     data = {
            'camera_id': cam_id,
            'camera_position':cam_pos,
            'density_count': int(count),
            'timestamp': str(timestamp),
            'density_map': density_map,
     }

     return data




def load_settings(self, location, file_name):
    try:
        json_file = open(location + '/' + file_name + '.txt')
    except IOError:
        print('IoError')
    else:
        line = json_file.readline()
        settings = json.loads(line)
        json_file.close()

        if 'model_path1' in settings:
            self.model_path1 = os.path.join(os.path.dirname(__file__), settings['model_path1'])
        if 'model_path2' in settings:
            self.model_path2 = os.path.join(os.path.dirname(__file__), settings['model_path2'])
        if 'scale' in settings:
            self.scale = settings['scale']
        if 'process_interval' in settings:
            self.process_interval = settings['process_interval']
        if 'save_on_count' in settings:
            self.save_on_count = settings['save_on_count']
        if 'mqtt' in settings:
            self.mqtt = settings['mqtt']
        if 'save_image_flag' in settings:
            self.save_image_flag = settings['save_image_flag']
    print('SETTINGS LOADED FOR MODULE: ' + self.module_id)
