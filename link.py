import cv2
import UtilityManager
import LogManager
from algos.poseEstimation import get_Pose
from algos.counting import get_crowd
from algos.flow_analysis import get_flow
from algos.fight.demo import fight
from algos.abnormal_behaviour.demo import abnormal
import numpy as np
import matplotlib.pyplot as plt

config = None


def set_configuration(InValue):
    global config
    config = InValue


# main processFrame function connects to RTSP and distributes frames/clips to algorithms
def processFrame(url, freq, vid_len):
    camera = cv2.VideoCapture(url)
    _, frame = camera.read()

    tempVideoWriter = UtilityManager.make_output_vid(frame, config.FRAMERATE, config.FRAMERATE)

    temp = []
    frameNo = 0
    previousFrame = []

    # infinite loop over the capture
    streamLoop = True
    while streamLoop:

        count, density_map = None, None
        pose = None
        fight_label = None
        abnormal_label = None
        flow_map, ave_flow_dir, ave_flow_mag = None, None, None
        tempFrameID = None
        frameId = camera.get(1)
        # Capture frame-by-frame
        ret, frame = camera.read()
        # frame = UtilityManager.resize_image(frame, InRatio=80)

        if ret:
            tempVideoWriter.write(frame)

        # Run the algorithm based on the given interval
        if frameId % (config.FRAMERATE * freq) == 0:
            tempVideoWriter = UtilityManager.make_output_vid(frame, config.FRAMERATE, frameId, tempVideoWriter)
            tempFrameID = temp
            temp = frameId
            frameNo = frameNo + 1

            if frameNo > 1:
                flow_map, ave_flow_mag, ave_flow_dir = get_flow.process_flow(frame, previousFrame)
                # count, density_map = get_crowd.process_crowd(frame)
                # pose = get_Pose.process_pose(frame)
            previousFrame = frame[:]

        if (frameNo > 0) and (frameNo % 2) == 0:
            #
            # fight_label = fight.process(config.TEMP_VIDEO_PATH, tempFrameID)
            # abnormal_label = abnormal.process(config.TEMP_VIDEO_PATH, tempFrameID)

            frameNo = frameNo - 1
            # streamLoop = False
            return frame, density_map, count, flow_map, ave_flow_mag, ave_flow_dir, pose, fight_label, abnormal_label

            # return frame, density_map, count, flow_map, ave_flow_dir, ave_flow_mag, fight_label, abnormal_label, pose


def run(InConfiguration):
    config = InConfiguration
    # UtilityManager.check_Camera(config.CAMERA_PATH)
    UtilityManager.displayTimeStame()

    fig = plt.figure()

    # Main Loop
    while True:

        ImgFromCamera, \
        density_map, count, \
        flow_map, ave_flow_mag, ave_flow_dir, \
        pose, fight_label, abnormal_label = processFrame(config.CAMERA_PATH, 5, 5)

        # ImgFromCamera, count, density_map, flow_map, pose = processFrame(config.CAMERA_PATH, 5, 5)
        UtilityManager.displayTimeStame()

        if count:
            LogManager.displayLog(f'Crowd Count:{count}', 'white', 'underline')
        if fight_label:
            LogManager.displayLog(f'Fight Detection Results : {fight_label}', 'white')
        if abnormal_label:
            LogManager.displayLog(f'Crowd abnormality Results :{abnormal_label} ', 'white')
        if ave_flow_dir:
            LogManager.displayLog(f'Ave flow direction = {ave_flow_dir}', 'white')
            LogManager.displayLog(f'Ave flow Magnitude  = {ave_flow_mag}', 'white')

        output = ImgFromCamera[:, :, ::-1]
        if density_map is not None:
            density_map = cv2.resize(density_map, (output.shape[1], output.shape[1]))
            density_map = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
            output = np.hstack(output, density_map)
            # plt.imshow(density_map)
        if flow_map is not None:
            # plt.imshow(flow_map[:, :, 0])
            flow = cv2.resize(flow_map[:, :, 0], (output.shape[1], output.shape[1]))
            output = np.hstack(output, flow)
        if pose is not None:
            # plt.imshow(pose)
            pose = cv2.resize(pose, (output.shape[1], output.shape[1]))
            output = np.hstack(output, pose)

        cv2.imwrite('/media/ramdisk/output.png', output)

        # fig.add_subplot(1, 3, 1)
        # plt.imshow(ImgFromCamera[:, :, ::-1])
        # if density_map is not None:
        #     fig.add_subplot(1, 3, 2)
        #     plt.imshow(density_map)
        # if flow_map is not None:
        #     fig.add_subplot(1, 3, 3)
        #     plt.imshow(flow_map[:, :, 0])
        # if pose is not None:
        #     fig.add_subplot(1, 3, 3)
        #     plt.imshow(pose)
        # plt.pause(0.001)
