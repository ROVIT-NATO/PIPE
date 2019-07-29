import cv2
import datetime
import os
import time
import math
from algos.counting import get_crowd
from algos.flow_analysis import get_flow
from algos.fight.demo import fight
from algos.abnormal_behaviour.demo import abnormal
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import matplotlib.pyplot as plt
import matplotlib.animation as animation




# Consts to be read from setting file

frameRate = 25

vidpath = (os.path.dirname(__file__) + '/algos/vid/')





# Video witer function

def make_output_vid(frame, frameId=None, out=None):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if out:
        out.release()


    return cv2.VideoWriter('algos/vid/' + str(frameId) + '.avi', fourcc, frameRate, (frame.shape[1], frame.shape[0]))


# main linker function connects to RTSP and distributes frames/clips to algorithms

def linker(url,freq,vid_len):


    cap = cv2.VideoCapture(url)
    frameRate = 25
    # if cap.isOpened():
    #     print("DRONE CONNECTION IS ESTABLISHED. PRESS Q TO TERMINATE")
    # else:
    #     print("OOPS!!! FAILED TO ESTABLISH CONNECTION. CHECK RTSP URL. PROCESS TERMINATED")

    _,frame = cap.read()

    out = make_output_vid(frame)

    temp=[]
    f = 0

    p_frame = []


    # infinite loop over the capture

    Round = True

    while Round:

        frameId = cap.get(1)

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            out.write(frame)

        # Run the algorithm based on the given interval

        if (frameId % (frameRate*freq) == 0):


            out = make_output_vid(frame, frameId, out)
            x=temp
            temp=frameId
            f = f+1



            # algorithms to be added here

            # print('----------------' + str(datetime.datetime.now()) + '------------------')




            # count, density_map = get_crowd.process_frame(frame)
            if (f>1):
                count, density_map = get_crowd.process_frame(frame)
                flow_map,ave_flow_mag,ave_flow_dir = get_flow.process_flow(frame,p_frame)

            p_frame=frame[:]

        if (f>0) and (f % 2) == 0:
            fight_label = fight.process(vidpath,x)
            abnormal_label = abnormal.process(vidpath,x)

            f=f-1

            # print('-------------------------------------------------------------')

            Round =False

            return frame,density_map,count,flow_map,ave_flow_dir,ave_flow_mag,fight_label,abnormal_label





cap = cv2.VideoCapture('rtsp://root:pass@10.144.129.107/axis-media/media.amp')
if cap.isOpened():
    print("DRONE CONNECTION IS ESTABLISHED. PRESS Q TO TERMINATE")
else:
    print("OOPS!!! FAILED TO ESTABLISH CONNECTION. CHECK RTSP URL. PROCESS TERMINATED")

print('--------------------' + str(datetime.datetime.now()) + '-----------------------')

f = plt.figure()

while True:


    ori,den_map,cnt,f_mp,flow_dir,flow_mag,fight_l,abnormal_l  = linker('rtsp://root:pass@10.144.129.107/axis-media/media.amp',5,5)

    print('--------------------' + str(datetime.datetime.now()) + '-----------------------')



    f.add_subplot(1, 3, 1)
    plt.imshow(den_map)
    f.add_subplot(1, 3, 2)
    plt.imshow(ori)
    f.add_subplot(1, 3, 3)
    plt.imshow(f_mp[:, :, 0])

    plt.pause(0.001)













