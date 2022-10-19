import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmdet.apis import inference_detector, init_detector, show_result_pyplot,show_result_ins

config='/content/SOLO/configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py'
checkpoint='/content/SOLO/epoch_12.pth'

model=init_detector(config, checkpoint, device='cuda:0')
print('Loaded')

img='/content/SOLO/test.png'
result=inference_detector(model,img)
show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="demo_out_6.jpg")