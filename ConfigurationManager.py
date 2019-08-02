import os


class get_Config:
    def __init__(self):
        self.CAMERA_PATH = 'rtsp://root:pass@10.144.129.107/axis-media/media.amp'
        self.FRAMERATE = 25
        self.VIDEO_FREQUENCY = 5
        self.VIDEO_LENGTH = 5
        self.TEMP_VIDEO_PATH = (os.path.dirname(__file__) + '/algos/vid/')
        self.GPU_ID = '0,1'
        self.IS_ENABLE_WARNING = False
        self.IS_RESIZE_INPUT_IMAGE = False
