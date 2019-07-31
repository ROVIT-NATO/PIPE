import os
import cv2



def set_CUDA_Environment(InGPU='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = InGPU




