from tqdm import tqdm
import torch
from pathlib import Path
from yolort.v5 import non_max_suppression
from yolort.v5.utils.torch_utils import time_sync
from yolort.v5.utils.metrics import ConfusionMatrix, ap_per_class
from yolort.v5.utils.val import process_batch
from yolort.v5.utils.dataloaders import *
from yolort.v5.utils.general import scale_coords


def cal_benchmark(
        data,
        batch_size,
        img_size,
        score_thresh,
        nms_thresh,
        model=None,
        loader=None,
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu,
        single_cls=None,
        is_v5=False,
):

    nc = 1 if single_cls else int(data['nc'])  # number of classes
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    half = False
    plots = False
    jdict, stats, ap, ap_class = [], [], [], []
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    pbar = tqdm(loader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    cuda = device.type != 'cpu'
    names = dict(enumerate(data['names']))

    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        t1 = time_sync()
        if cuda:
            im = im.to(device)  # im  待检测的图片
            targets = targets.to(device)  # target 是标签么

        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        with torch.no_grad():
            yolo_dets = model(im)
            dt[1] += time_sync() - t2
            if is_v5:
                t3 = time_sync()
                yolo_dets = non_max_suppression(yolo_dets, score_thresh, nms_thresh, agnostic=False)
                dt[2] += time_sync() - t3

        # Metrics
        for si, pred in enumerate(yolo_dets):

            labels = targets[targets[:, 0] == si, 1:]  # 这个batch 中的检测框信息  5维度
            nl, npr = labels.shape[0], len(pred)  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            # if npr == 0:
            #     if nl:
            #         stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
            #         if plots:
            #             confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
            #     continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0

            # detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            if is_v5:
                predn = pred.clone()  # predn = pred["boxes"].clone()
                scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pre
            else:
                predn = pred.copy()  # predn = pred["boxes"].clone()
                scale_coords(im[si].shape[1:], predn["boxes"], shape, shapes[si][1])  # native-space pre

            # Evaluate
            if nl:
                tbox = xywhn2xyxy(labels[:, 1:5])  # target boxes   # xy xy    目标框  这里没问题
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels  拼接 标签和检测框
                if not is_v5:
                    predn = torch.cat((predn["boxes"], predn["scores"].unsqueeze(-1), predn["labels"].unsqueeze(-1)), 1)
                correct = process_batch(predn, labelsn, iouv)
            if is_v5:
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
            else:
                stats.append((correct, pred['scores'], pred['labels'], labels[:, 0]))  # (correct, conf, pcls, tcls)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().detach().numpy() for x in zip(*stats)]  # to numpy  stats(list)  x(tuple)
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=None, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    shape = (batch_size, 3, img_size, img_size)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
