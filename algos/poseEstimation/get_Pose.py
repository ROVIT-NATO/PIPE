import algos.poseEstimation.TestManager as TestManager
import os
import cv2


session = TestManager.init(InCheckPointPath=os.path.dirname(__file__) + '/checkpoints/train/',
                           vgg19_path=os.path.dirname(__file__) + '/checkpoints/vgg/vgg_19.ckpt')


def process_pose(InFrame):
    # convert BGR to RGB
    return cv2.cvtColor(TestManager.processFrame(InFrame, session),cv2.COLOR_BGR2RGB)

# if __name__ == '__main__':
#     # session = TestManager.init(InCheckPointPath='checkpoints/train/',
#     #                            vgg19_path='checkpoints/vgg/vgg_19.ckpt')
#
#     while True:
#         stream = cv2.VideoCapture('rtsp://root:pass@10.144.129.107/axis-media/media.amp')
#         ret, frame = stream.read()
#         process_pose(frame)
