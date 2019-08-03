import numpy as np
import sys
import os

from algos.abnormal_behaviour.keras_video_classifier.library.recurrent_networks import \
    VGG16BidirectionalLSTMVideoClassifier

vgg16_include_top = True

model_dir_path = os.path.join(os.path.dirname(__file__), 'models', 'UCF-101')
config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
                                                                              vgg16_include_top=vgg16_include_top)
weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
                                                                              vgg16_include_top=vgg16_include_top)

predictor = VGG16BidirectionalLSTMVideoClassifier()
predictor.load_model(config_file_path, weight_file_path)


def process(vidpath, frameId):
    videoFile = f'{vidpath}/{frameId}.avi'
    predicted_label = predictor.predict(videoFile)

    os.remove(videoFile)

    return predicted_label
