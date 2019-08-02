import os


class get_Config:
    def __init__(self):
        self.CAMERA_PATH = 'rtsp://root:pass@10.144.129.107/axis-media/media.amp'
        self.FRAMERATE = 25
        self.VIDEO_FREQUENCY = 5
        self.VIDEO_LENGTH = 5
        self.TEMP_VIDEO_PATH = (os.path.dirname(__file__) + '/algos/vid/')
        self.GPUID='0,1'




