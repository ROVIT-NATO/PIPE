import os
import datetime
import warnings
import cv2
import sys
import LogManager
import shutil


def set_CUDA_Environment(InGPU='0'):
    os.environ['CUDA_VISIBLE_DEVICES'] = InGPU


def make_output_vid(InFrame, InFrameRate, InVideoPath='algos/vid/', InFrameID=None, InVideoWriter=None ):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if InVideoWriter:
        InVideoWriter.release()

    return cv2.VideoWriter(f'{InVideoPath}/{InFrameID}.avi',
                           fourcc,
                           InFrameRate,
                           (InFrame.shape[1], InFrame.shape[0]))


def Is_File_Exist(InPath):
    if os.path.isfile(InPath):
        return True
    else:
        return False


def check_Camera(InPath):
    # cap = cv2.VideoCapture('rtsp://root:pass@10.144.129.107/axis-media/media.amp')
    stream = cv2.VideoCapture(InPath)

    if stream.isOpened():
        LogManager.displayLog(f'[Info] Loading {InPath}', 'blue')
        LogManager.displayLog("[Info] Drone connection is established.", 'green')
    else:
        LogManager.displayLog("[Failed] Failed to establish connection. Check RTSP URL. Process Terminate", 'red')
        sys.exit(-1)

    stream.release()


def enable_Warning(InValue=False):
    if InValue is False:
        LogManager.displayLog('[Info] Warning message disable.')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        warnings.filterwarnings('ignore')


def displayTimeStame():
    LogManager.displayLog(f'--------------------{datetime.datetime.now()}-----------------------')


def resize_image(InImage, InRatio):
    # calculate the 50 percent of original dimensions
    width = int(InImage.shape[1] * InRatio / 100)
    height = int(InImage.shape[0] * InRatio / 100)
    # dsize
    dsize = (width, height)
    # resize image
    return cv2.resize(InImage, dsize)


def create_Folder(InPath):
    if not os.path.exists(InPath):
        os.makedirs(InPath)
        LogManager.displayLog(f'[Info] Folder {InPath} is created', 'blue')
    else:
        LogManager.displayLog(f'[Info] Folder {InPath} already exist.','red')


def remove_Folder(InPath):
    LogManager.displayLog(f'[Info] Cleaning temp folder {InPath}')
    try:
        shutil.rmtree(InPath)
        return True
    except:
        LogManager.displayLog(f'[Info] Folder {InPath} not found!')
        return False