import cv2
import numpy as np
from torch.autograd import Variable
import torch
import math
import os

# from algos.flow_analysis.FlowNet2_src import flow_to_image
from algos.flow_analysis.FlowNet2_src import FlowNet2

model = []
path = os.path.dirname(__file__) + '/FlowNet2_src/pretrained/FlowNet2_checkpoint.pth.tar'

flownet2 = FlowNet2()

pretrained_dict = torch.load(path)['state_dict']

model_dict = flownet2.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
flownet2.load_state_dict(model_dict)

flownet2.cuda()

model = flownet2


def process_flow(frame, p_frame):
    height, width = frame.shape[:2]

    fr1 = cv2.resize(frame, (384, 512))
    fr2 = cv2.resize(p_frame, (384, 512))

    ims = np.array([[fr1, fr2]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)
    ims = torch.from_numpy(ims)
    ims_v = Variable(ims.cuda(), requires_grad=False)

    flownet_2 = model
    flow_uv = flownet_2(ims_v).cpu().data
    flow_uv = flow_uv[0].numpy().transpose((1, 2, 0))

    # # CONVERT BACK TO ORIGINAL SCALE
    flow_uv = cv2.resize(flow_uv, (width, height))

    ave_flow_mag = []
    ave_flow_dir = []

    flow_uv_current = flow_uv

    mean_u = flow_uv_current[:, :, 0].mean()
    mean_v = flow_uv_current[:, :, 1].mean()

    mag = math.sqrt(math.pow(mean_u, 2) + math.pow(mean_v, 2))

    if mean_v < 0:
        uv_angle = 360 + math.degrees(math.atan2(mean_v, mean_u))
    else:
        uv_angle = math.degrees(math.atan2(mean_v, mean_u))
    direction = uv_angle / 360

    ave_flow_mag.append(mag)
    ave_flow_dir.append(direction)
    #
    # print('Ave flow direction = ', ave_flow_dir)
    # print('Ave flow Magnitude  = ', ave_flow_mag)

    return flow_uv, ave_flow_mag, ave_flow_dir
