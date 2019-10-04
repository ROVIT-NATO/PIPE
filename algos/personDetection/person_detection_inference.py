
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
import PyQt5
import matplotlib
matplotlib.use('qt5agg')
from matplotlib import pyplot as plt
import os
import image_slicer
from image_slicer import join
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import cv2
import scipy
import scipy.misc
from scipy.misc import imsave
from scipy import ndimage
from scipy import misc

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

# Set the image size.
img_height = 600
img_width = 600

# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=1,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)



# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = './manzoor_models/Person/ssd300_Person-34_loss-3.3037_val_loss-3.5082.h5'
# weights_path = '/ocean/manzoor/models_h5/ssd300_pascal_07+12_102k_steps.h5'
model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.
#
# # We'll only load one image in this example.
img_path = '/ocean/manzoor/Person_images/images/001.jpg'
#
# #rootdir = './manzoor_models/Person/images_test_split/'
#
# # for subdir, dirs, files in os.walk(rootdir):
# #     for file in files:
# #         img_path = subdir + '/' + file
# #         print(img_path)
# #
# #         orig_images = []  # Store the images here.
# #         input_images = []  # Store resized versions of the images here.
#

orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img)
input_images.append(img)
input_images = np.array(input_images)

y_pred = model.predict(input_images)

confidence_threshold = 0.5

y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh[0])

# Visualize predictions
# Display the image and draw the predicted boxes onto it.

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
classes = ['background',
           'Person']
# classes = ['background',
#            'aeroplane', 'bicycle', 'sheep', 'boat',
#            'bottle', 'bus', 'car', 'cat',
#            'chair', 'cow', 'diningtable', 'dog',
#            'horse', 'motorbike', 'person', 'pottedplant',
#            'sheepss', 'sofa', 'train', 'tvmonitor']

plt.figure(figsize=(20, 12))
plt.imshow(orig_images[0])

current_axis = plt.gca()

for box in y_pred_thresh[0]:
    # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
    xmin = box[2] * orig_images[0].shape[1] / img_width
    ymin = box[3] * orig_images[0].shape[0] / img_height
    xmax = box[4] * orig_images[0].shape[1] / img_width
    ymax = box[5] * orig_images[0].shape[0] / img_height
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(
        plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})

plt.show()


#
# ####   split and try ####
#
# image_path = '/ocean/manzoor/Person_images/images/5554.jpg'
#
# num_tiles = 40
# tiles = image_slicer.slice(image_path, num_tiles)
#
# for tile in tiles:
#     orig_images = []  # Store the images here.
#     input_images = []  # Store resized versions of the images here.
#
#     source_img = Image.open(tile.filename).convert("RGBA")
#     img_path = tile.filename
#     im = scipy.misc.imread(tile.filename)
#     orig_images.append(imread(img_path))
#     img = image.load_img(img_path, target_size=(img_height, img_width))
#     img = image.img_to_array(img)
#     input_images.append(img)
#     input_images = np.array(input_images)
#
#     y_pred = model.predict(input_images)
#
#     confidence_threshold = 0.5
#
#     y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
#
#     np.set_printoptions(precision=2, suppress=True, linewidth=90)
#     print("Predicted boxes:\n")
#     print('   class   conf xmin   ymin   xmax   ymax')
#     print(y_pred_thresh[0])
#
#     # Visualize predictions
#     # Display the image and draw the predicted boxes onto it.
#
#     # Set the colors for the bounding boxes
#     # fig = plt.figure()
#     colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
#     classes = ['background',
#                'Sheep']
#     # classes = ['background',
#     #            'aeroplane', 'bicycle', 'sheep', 'boat',
#     #            'bottle', 'bus', 'car', 'cat',
#     #            'chair', 'cow', 'diningtable', 'dog',
#     #            'horse', 'motorbike', 'person', 'pottedplant',
#     #            'sheepss', 'sofa', 'train', 'tvmonitor']
#
#     #plt.figure(figsize=(20, 12))
#     #plt.imshow(orig_images[0])
#     width = 4
#
#
#     current_axis = plt.gca()
#
#     for box in y_pred_thresh[0]:
#         # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
#         xmin = box[2] * orig_images[0].shape[1] / img_width
#         ymin = box[3] * orig_images[0].shape[0] / img_height
#         xmax = box[4] * orig_images[0].shape[1] / img_width
#         ymax = box[5] * orig_images[0].shape[0] / img_height
#         color = colors[int(box[0])]
#         label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#         current_axis.add_patch(
#             plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=1))
#         current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
#
#
#
#
#         draw = ImageDraw.Draw(source_img)
#         for i in range(width):
#             draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=None, outline="red")
#             xmin = xmin + 1
#             ymin = ymin + 1
#             xmax = xmax + 1
#             ymax = ymax + 1
#
#         #draw.text((20, 70), "something123", font=ImageFont.truetype("font_path123"))
#         #im_arr_bgr = cv2.cvtColor(input_images, cv2.COLOR_RGB2BGR)
#         #cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
#         #cv2.imwrite("my.png", im)
#
#     source_img.save(tile.filename)
#
#     # fig1 = plt.gcf()
#     # plt.show()
#     #plt.savefig(tile.filename, bbox_inches='tight', pad_inches=0, transparent=True)
#     tile.image = Image.open(tile.filename)
#
# joinedImage = join(tiles)
# joinedImage.save('')

