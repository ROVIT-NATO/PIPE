import os
import datetime
import warnings
import cv2
import sys
import LogManager


def set_CUDA_Environment(InGPU='0'):
    os.environ['CUDA_VISIBLE_DEVICES'] = InGPU


def make_output_vid(InFrame, InFrameRate, InFrameID=None, videoWriter=None):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if videoWriter:
        videoWriter.release()
    return cv2.VideoWriter(f'algos/vid/{InFrameID}.avi',
                           fourcc,
                           InFrameRate,
                           (InFrame.shape[1], InFrame.shape[0]))


def isFileExist(InPath):
    if os.path.isfile(InPath):
        return True
    else:
        False


def check_Camera(InPath):
    # cap = cv2.VideoCapture('rtsp://root:pass@10.144.129.107/axis-media/media.amp')
    stream = cv2.VideoCapture(InPath)

    if stream.isOpened():
        LogManager.displayLog(f'[Info] Loading {InPath}', 'blue')
        LogManager.displayLog("[Info] Drone connection is established.", 'green')
    else:
        LogManager.displayLog("[Failed] Failed to establish connection. Check RTSP URL. Process Terminate", 'red')
        sys.exit(-1)


def ignore_Warning(InValue=True):
    if InValue:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        warnings.filterwarnings('ignore')


def displayTimeStame():
    LogManager.displayLog('--------------------' + str(datetime.datetime.now()) + '-----------------------')


def resize_image(InImage, InRatio):
    # calculate the 50 percent of original dimensions
    width = int(InImage.shape[1] * InRatio / 100)
    height = int(InImage.shape[0] * InRatio / 100)
    # dsize
    dsize = (width, height)
    # resize image
    return cv2.resize(InImage, dsize)
