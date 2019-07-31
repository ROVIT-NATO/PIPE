import TestManager
import os
import cv2
import warnings

warnings.filterwarnings('ignore')

session = TestManager.init(InCheckPointPath=os.path.dirname(__file__) + '/checkpoints/train/',
                           vgg19_path=os.path.dirname(__file__) + '/checkpoints/vgg/vgg_19.ckpt')


def get_Pose(InFrame):
    return TestManager.processFrame(InFrame, session)

# if __name__ == '__main__':
#     # session = TestManager.init(InCheckPointPath='checkpoints/train/',
#     #                            vgg19_path='checkpoints/vgg/vgg_19.ckpt')
#
#     while True:
#         stream = cv2.VideoCapture('rtsp://root:pass@10.144.129.107/axis-media/media.amp')
#         ret, frame = stream.read()
#         get_Pose(frame)
