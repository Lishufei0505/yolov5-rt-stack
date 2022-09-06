import yolort

import argparse
import platform
import sys
import time
from pathlib import Path

import pandas as pd

FILE = Path().resolve()
ROOT = FILE.parents[0]  # YOLOrt root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import os
import cv2
import numpy

import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # PCI_BUS_ID”
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = device.type != 'cpu'

from yolort.models.yolo import YOLO
from yolort.models import YOLOv5
from yolort.utils import Visualizer, get_image_from_url, read_image_to_tensor, check_dataset
from yolort.v5 import load_yolov5_model, letterbox, non_max_suppression, attempt_download
from yolort.v5.utils.downloads import safe_download
from yolort.v5.utils.dataloaders import *
from yolort.v5.utils.general import colorstr, increment_path


from yolort.data.metric import cal_benchmark



data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
data = check_dataset(data)

img_size = 640
stride = 32
score_thresh = 0.25
nms_thresh = 0.45
batch_size = 32

single_cls=False # treat as single-class dataset
rect = False
workers=8  # max dataloader workers (per RANK in DDP mode)


task = 'val'
task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
loader,dataset = create_dataloader(data[task],
                               img_size,
                               stride,
                               batch_size,
                               single_cls,
                               rect=rect,
                               workers=workers,
                               prefix=colorstr(f'{task}: '))

model_path = 'yolov5s.pt'
checkpoint_path = attempt_download(model_path)

# load yolov5 model
model_yolov5 = load_yolov5_model(checkpoint_path)
model_yolov5 = model_yolov5.to(device)
model_yolov5 = model_yolov5.eval()

cal_benchmark(data,
              batch_size,
              img_size,
              score_thresh,
              nms_thresh,
              model_yolov5,
              loader,
              device,
              single_cls,
              is_v5=True
             )


# load yolort model
model_yolort = YOLO.load_from_yolov5(
    checkpoint_path,
    score_thresh=score_thresh,
    nms_thresh=nms_thresh,
    version="r6.0",
)
model_yolort = model_yolort.eval()
model_yolort = model_yolort.to(device)


cal_benchmark(data,
              batch_size,
              img_size,
              score_thresh,
              nms_thresh,
              model_yolort,
              loader,
              device,
              single_cls,
              is_v5=False
             )