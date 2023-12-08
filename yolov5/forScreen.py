#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2023 - 2023 Gosu, Inc. All Rights Reserved 
#
# @Time    : 2023/12/8 10:06
# @Author  : GosuXX
# @File    : forScreen.py
import random
from copy import deepcopy

import cv2
import numpy as np
import pyautogui
import torch
import wx
# 加载模型
import yaml
from prettytable import PrettyTable
from ultralytics.utils.torch_utils import select_device

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression


def mouse_snip(x, y):
    pyautogui.moveTo(x, y)
    pyautogui.leftClick(x, y)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]  # random color
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # coordinates
    cv2.rectangle(img, c1, c2, color=color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


device = select_device('')
dataset_dict = yaml.load(open("data/coco128.yaml"), yaml.FullLoader)
model = attempt_load('runs/train/exp20231205/weights/best.pt')  # 加载FP32模型
stride = int(model.stride.max())  # 模型步长
names = model.module.names if hasattr(model, 'module') else model.names  # 获取类名

app = wx.App()
dc = wx.ScreenDC()
dc.SetBrush(wx.TRANSPARENT_BRUSH)
dc.SetPen(wx.Pen((255, 0, 0), width=10, style=wx.PENSTYLE_SOLID))

while True:
    screen = pyautogui.screenshot()
    screen_np = np.array(screen)
    pred_img = deepcopy(screen_np)
    high, width = screen_np.shape[:2]
    high_ratio, width_ratio = high / 640, width / 640
    screen_np = letterbox(screen_np, new_shape=640)[0]
    # pred_img = deepcopy(screen_np)
    screen_np = screen_np[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    screen_np = np.ascontiguousarray(screen_np)
    screen_np = torch.from_numpy(screen_np).to(device)
    screen_np = screen_np.float() / 255.0  # 0 - 255 to 0.0 - 1.0
    if screen_np.ndimension() == 3:
        screen_np = screen_np.unsqueeze(0)
    # 推理
    pred = model(screen_np, augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
    tb = PrettyTable(field_names=["Class", "Conf", "Point<x1 x2 y1 y2>"])
    for res in pred:
        for *box, conf, cls in res:
            box[0] *= width_ratio
            box[2] *= width_ratio
            box[1] *= high_ratio
            box[3] *= high_ratio
            x1, y1, x2, y2 = map(int, box)
            tb.add_row([dataset_dict["names"][int(cls)],
                        conf, ",".join(map(lambda x: str(x), [x1, x2, y1, y2]))])
            label = f"{dataset_dict['names'][int(cls)]}:{format(conf, '.3f')}"
            plot_one_box((x1, y1, x2, y2), pred_img, label=label, color=(0, 0, 255), line_thickness=2)
            # dc.DrawRectangle(x1, y1, x2, y2)
            mouse_snip((x1 + x2) / 2, (y1 + y2))
    print(tb)
    # 显示图像
    cv2.namedWindow('Screen Capture', cv2.WINDOW_NORMAL)
    cv2.imshow('Screen Capture', pred_img)
    cv2.waitKey(1)
