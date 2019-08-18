import cv2
import UtilityManager
import LogManager
import numpy as np

LogManager.displayLog('[Info] Loading Pose Detection ...', 'blue')
from algos.poseEstimation import get_Pose

#
LogManager.displayLog('[Info] Loading Crowd Counting  ...', 'blue')
from algos.counting import get_crowd

LogManager.displayLog('[Info] Loading Flow Detection  ...', 'blue')
from algos.flow_analysis import get_flow

LogManager.displayLog('[Info] Loading Fight Detection ...', 'blue')
from algos.fight.demo import fight

LogManager.displayLog('[Info] Loading Abnormal Behaviour Detection ...', 'blue')
from algos.abnormal_behaviour.demo import abnormal

# import matplotlib.pyplot as plt

config = None
import GUIManager


def set_configuration(InValue):
    global config
    config = InValue


# main processFrame function connects to RTSP and distributes frames/clips to algorithms
def processFrame(url, freq):
    camera = cv2.VideoCapture(url)
    ret, frame = camera.read()
    if ret is False:
        LogManager.displayLog('[Error] Could not load camera!', 'red')

    tempVideoWriter = UtilityManager.make_output_vid(InFrame=frame,
                                                     InFrameRate=config.FRAMERATE,
                                                     InVideoPath=config.TEMP_VIDEO_PATH,
                                                     InFrameID=None,
                                                     InVideoWriter=None)

    temp = []
    frameNo = 0
    previousFrame = []

    # infinite loop over the capture
    streamLoop = True
    while streamLoop:

        count, density_map = 'Processing .. ', None
        pose = None
        fight_label = 'Processing .. '
        abnormal_label = 'Processing .. '
        flow_map, ave_flow_dir, ave_flow_mag = None, 'Processing .. ', 'Processing .. '
        tempFrameID = None

        frameId = camera.get(1)
        # Capture frame-by-frame
        ret, frame = camera.read()
        # resize the image 50%
        if config.IS_RESIZE_INPUT_IMAGE:
            frame = UtilityManager.resize_image(frame, InRatio=50)

        if ret:
            tempVideoWriter.write(frame)

        # Run the algorithm based on the given interval
        if frameId % (config.FRAMERATE * freq) == 0:
            tempVideoWriter = UtilityManager.make_output_vid(InFrame=frame,
                                                             InFrameRate=config.FRAMERATE,
                                                             InVideoPath=config.TEMP_VIDEO_PATH,
                                                             InFrameID=frameId,
                                                             InVideoWriter=tempVideoWriter)
            tempFrameID = temp
            temp = frameId
            frameNo = frameNo + 1

            if frameNo > 1:
                flow_map, ave_flow_mag, ave_flow_dir = get_flow.process_flow(frame, previousFrame)
                count, density_map = get_crowd.process_crowd(frame)
                pose = get_Pose.process_pose(frame)
            previousFrame = frame[:]

        if (frameNo > 0) and (frameNo % 2) == 0:
            fight_label = fight.process(config.TEMP_VIDEO_PATH, tempFrameID)
            abnormal_label = abnormal.process(config.TEMP_VIDEO_PATH, tempFrameID)

            frameNo = frameNo - 1
            return frame, \
                   density_map, count, \
                   flow_map, np.squeeze(ave_flow_mag), np.squeeze(ave_flow_dir), \
                   pose, \
                   fight_label, \
                   abnormal_label


def run():
    UtilityManager.remove_Folder(config.TEMP_VIDEO_PATH)
    UtilityManager.create_Folder(config.TEMP_VIDEO_PATH)

    UtilityManager.displayTimeStame()

    # ImgFromCamera= np.zeros((256,256))
    # density_map = np.zeros((256, 256))
    # pose = np.zeros((256, 256))
    # flow_map = np.zeros((256, 256))
    # count =0
    # fight_label= 'noFight'
    # abnormal_label= 'low'
    # ave_flow_dir, ave_flow_mag =0,0

    window = GUIManager.get_window()
    window.create_plot(InFigureSize=(10, 10), InColumns=2, InRows=2, InTitle='Kingston University')

    while True:
        ImgFromCamera, \
        density_map, count, \
        flow_map, ave_flow_mag, ave_flow_dir, \
        pose, \
        fight_label, \
        abnormal_label = processFrame(config.CAMERA_PATH, 5)

        UtilityManager.displayTimeStame()
        window.add_sub_plot(cv2.cvtColor(ImgFromCamera, cv2.COLOR_BGR2RGB), 1, 'Drone View')
        window.add_sub_plot(density_map, 2, 'Density Estimation')
        window.add_sub_plot(pose, 3, 'Pose Estimation')
        window.add_sub_plot(flow_map, 4, 'Flow Estimation')

        window.add_text(f'Density count : {count}', InXPos=-500, InYPos=300, InColor='blue')
        if fight_label == 'noFight':
            window.add_text(f'Fight Detection : {fight_label}', InXPos=-500, InYPos=325, InColor='green')
        else:
            window.add_text(f'Fight Detection : {fight_label}', InXPos=-500, InYPos=325, InColor='red')
        if abnormal_label == 'low':
            window.add_text(f'Crowd abnormality : {abnormal_label} ', InXPos=-200, InYPos=315, InColor='green')
        else:
            window.add_text(f'Crowd abnormality : {abnormal_label} ', InXPos=-200, InYPos=315, InColor='red')
        window.add_text(f'Ave flow direction : {np.around(ave_flow_dir,5)}', InXPos=120, InYPos=300, InColor='brown')
        window.add_text(f'Ave flow Magnitude : {np.around(ave_flow_mag,5)}', InXPos=120, InYPos=320, InColor='purple')
        window.show()




