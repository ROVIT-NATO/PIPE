import cv2
import sys
# import Settings
# from algos.counting import get_crowd
from algos.flow_analysis import get_flow
# from algos.fight.demo import fight
# from algos.abnormal_behaviour.demo import abnormal
from algos.poseEstimation.Main import process_pose


def checkCamera(InPath):
    # cap = cv2.VideoCapture('rtsp://root:pass@10.144.129.107/axis-media/media.amp')
    stream = cv2.VideoCapture(InPath)

    if stream.isOpened():
        print("DRONE CONNECTION IS ESTABLISHED. PRESS Q TO TERMINATE")
    else:
        print("OOPS!!! FAILED TO ESTABLISH CONNECTION. CHECK RTSP URL. PROCESS TERMINATED")
        sys.exit(-1)


def make_output_vid(InFrame, InFrameRate, InFrameID=None, InTempVideoWriter=None):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if InTempVideoWriter is True:
        InTempVideoWriter.release()

    return cv2.VideoWriter('algos/vid/' + str(InFrameID) + '.avi', fourcc, InFrameRate,
                           (InFrame.shape[1], InFrame.shape[0]))


def stream_Loop(InSettings, InIsStreamLoop=True):
    # init the variable

    frameNo = 0
    p_frame = []
    temp = []
    count, density_map = None, None
    flow_map, ave_flow_mag, ave_flow_dir = None, None, None
    fight_label = None
    abnormal_label = None

    capture = cv2.VideoCapture(InSettings.CAMERA_PATH)
    _, frame = capture.read()

    tempVideoWriter = make_output_vid(frame, InSettings.FRAMERATE)

    while InIsStreamLoop:
        frameId = capture.get(1)
        ret, frame = capture.read()

        if ret:
            tempVideoWriter.write(frame)

        if frameId % (InSettings.FRAMERATE * InSettings.VIDEO_FREQUENCY) == 0:
            tempVideoWriter = make_output_vid(frame, InSettings.FRAMERATE, frameId, tempVideoWriter)

            x = temp
            temp = frameId
            frameNo = frameNo + 1

            if frameNo > 1:
                process_pose(frame)
                # count, density_map = get_crowd.process_pose(frame)
                # flow_map, ave_flow_mag, ave_flow_dir = get_flow.process_flow(frame, p_frame)

                p_frame = frame[:]

            if frameNo > 0 and frameNo % 2 == 0:
                # fight_label = fight.process(InSettings.TEMP_VIDEO_PATH, x)
                # abnormal_label = abnormal.process(InSettings.TEMP_VIDEO_PATH, x)

                frameNo = frameNo - 1

            InIsStreamLoop = False
            return frame, density_map, count, flow_map, ave_flow_dir, ave_flow_mag, fight_label, abnormal_label


def run(InSettings):
    checkCamera(InSettings.CAMERA_PATH)
    while True:
        ori, den_map, cnt, f_mp, flow_dir, \
        flow_mag, fight_l, abnormal_l = stream_Loop(InSettings)
