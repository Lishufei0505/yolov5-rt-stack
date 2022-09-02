import yolort

print(yolort.__version__)
print(yolort.__file__)
# %%
import argparse
import platform
import sys
import time
from pathlib import Path
# %%
import pandas as pd

# %%
FILE = Path().resolve()
ROOT = FILE.parents[0]  # YOLOrt root directory
print(str(ROOT))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
print(str(ROOT) in sys.path)
# %% md
### 配置环境
# %%
import os
import cv2
import numpy

import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # PCI_BUS_ID” # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = device.type != 'cpu'
# %% md
### 导入模型和定义的函数
# %%
from yolort.models.yolo import YOLO
from yolort.utils import Visualizer, get_image_from_url, read_image_to_tensor, check_dataset
from yolort.v5 import load_yolov5_model, letterbox, non_max_suppression, scale_coords, attempt_download
from yolort.v5.utils.downloads import safe_download
from yolort.v5.utils.dataloaders import *
from yolort.v5.utils.general import colorstr, increment_path
from yolort.v5.utils.torch_utils import time_sync
from yolort.v5.utils.metrics import ConfusionMatrix, ap_per_class
from yolort.v5.utils.val import process_batch

# %% md
### 加载数据
# %%
data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
data = check_dataset(data)
# %%
print(data)
# %%
# img_source = "https://huggingface.co/spaces/zhiqwang/assets/resolve/main/bus.jpg"
# # img_source = "https://huggingface.co/spaces/zhiqwang/assets/resolve/main/zidane.jpg"
# img_raw = get_image_from_url(img_source)
# %%
# print(img_raw)
# %%
# image = letterbox(img_raw, new_shape=(img_size, img_size), stride=stride)[0]
# image = read_image_to_tensor(image)
# image = image.to(device)
# image = image[None]
# print(image.size())
# %%
img_size = 640
stride = 64
score_thresh = 0.25
batch_size = 32
nms_thresh = 0.45
single_cls = False  # treat as single-class dataset
rect = False
workers = 8  # max dataloader workers (per RANK in DDP mode)
# %%
# print(batch_size)
# %%
# images_source =
# 我现在需要做的是不是把images/train2017的图片加载进来？
task = 'val'
task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
loader, dataset = create_dataloader(data[task],
                                    img_size,
                                    stride,
                                    batch_size,
                                    single_cls,
                                    rect=rect,
                                    workers=workers,
                                    prefix=colorstr(f'{task}: '))

# %%
# print(len(loader)) #loader 的长度是4 128/32
# print(len(dataset))  # 数据集的长度是128
# %%
nc = 1 if single_cls else int(data['nc'])  # number of classes
seen = 0
confusion_matrix = ConfusionMatrix(nc=nc)



# %% md
### 加载模型权重
# %%
# 如果项目路径下没有的话去指定的路径下下载
model_path = 'yolov5s.pt'
checkpoint_path = attempt_download(model_path)
# %% md
### 加载yolov5模型
# %%
# model_yolov5 = load_yolov5_model(checkpoint_path, fuse=True)
# model_yolov5 = model_yolov5.to(device)
# model_yolov5 = model_yolov5.eval()
# %%
# print(model_yolov5.names)  # 也可以用 data['names'] 代替
# %% md
### 加载yolort模型
# %%
model_yolort = YOLO.load_from_yolov5(
    checkpoint_path,
    score_thresh=score_thresh,
    nms_thresh=nms_thresh,
    version="r6.0",
)
model_yolort = model_yolort.eval()
model_yolort = model_yolort.to(device)


# %%
project = ROOT / 'runs/val',  # save to project/name
print(project)
name = 'exp',  # save to project/name
print(type(data['names']), data['names'])
names = dict(enumerate(data['names']))
print(type(names), names)
# Directories
# save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
# (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
# %%
# half &= device.type != 'cpu'  # FP16  # FP16 supported on limited backends with CUDA
half = False
plots = False

jdict, stats, ap, ap_class = [], [], [], []
s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
# print(s)
pbar = tqdm(loader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
# print(type(pbar))
dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
niou = iouv.numel()

for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
    #     print("batch: ", batch_i)
    t1 = time_sync()
    #     print("----------1111----------")
    #     print(shapes[0])
    #     print("shapes:",len(shapes[0]))
    #     print(targets.size())
    if cuda:
        im = im.to(device)  # im  待检测的图片
        targets = targets.to(device)  # target 是标签么
    #         print("targets:",targets)
    #         print("paths:", paths)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    nb, _, height, width = im.shape  # batch size, channels, height, width
    t2 = time_sync()
    dt[0] += t2 - t1
    #     print("--------------------22222-----------------------")
    #     print("shapes:",shapes)

    yolort_dets = model_yolort(im)

    #     print(yolort_dets)
    # Metrics
    for si, pred in enumerate(yolort_dets):

        labels = targets[targets[:, 0] == si, 1:]  # 这个batch 中的检测框信息  5维度


        nl, npr = labels.shape[0], len(pred)  # number of labels, predictions
        path, shape = Path(paths[si]), shapes[si][0]
        correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
        seen += 1

        if npr == 0:
            if nl:
                stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                if plots:
                    confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
            continue

        # Predictions
        if single_cls:
            pred[:, 5] = 0
        #         predn = pred["boxes"].clone()
        # detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        predn = pred.copy()

        scale_coords(im[si].shape[1:], predn["boxes"], shape, shapes[si][1])  # native-space pre

        # Evaluate
        if nl:
            tbox = xywhn2xyxy(labels[:, 1:5])  # target boxes   # xy xy    目标框  这里没问题

            tbox1 = labels[:, 1:5]  # xy wh  格式

            scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
            # 这里出问题了，传进去的还是正常的，出来了之后就有两列变成了0

            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels  拼接 标签和检测框


            predn = torch.cat((predn["boxes"], predn["scores"].unsqueeze(-1), predn["labels"].unsqueeze(-1)), 1)
            #             print("拼接后的结果：")
            #             print("predn", predn)
            #             print("labelsn:", labelsn)
            correct = process_batch(predn, labelsn, iouv)
        #             print("correct：", correct)
        #             if plots:
        #                 confusion_matrix.process_batch(predn, labelsn)
        stats.append((correct, pred['scores'], pred['labels'], labels[:, 0]))  # (correct, conf, pcls, tcls)
    #         print("stats:", type(stats))   stats 是list 类型的

    #     for x in zip(*stats):
    #         print("xxxx:", type(x),x)  #tuple  元组类型


stats = [torch.from_numpy(torch.cat(x, 0).cpu().detach().numpy()) for x in zip(*stats)]  # to numpy

if len(stats) and stats[0].any():
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=None, names=names)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
nt = np.bincount(int(stats[3]), minlength=nc)  # number of targets per class

# Print results
pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))