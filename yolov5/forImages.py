#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2023 - 2023 Gosu, Inc. All Rights Reserved 
#
# @Time    : 2023/12/4 15:03
# @Author  : GosuXX
# @File    : forImages.py
import multiprocessing as mp
import os
from copy import deepcopy
from queue import Queue
from threading import Thread

import cv2
import torch
import yaml
from prettytable import PrettyTable

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from utils.torch_utils import select_device

save_dir = "runs/user"
img_dir = "data/images"
dataset_dict = yaml.load(open("data/coco128.yaml"), yaml.FullLoader)
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model = attempt_load('runs/train/exp20231205/weights/best.pt')
pred_queue = Queue(20)


def read_file_pretreatment(file, name):
    try:
        img = cv2.imread(file)
        img = letterbox(img, new_shape=(640, 640))[0]
        pred_img = deepcopy(img)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = img.copy()  # make a copy of the array
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = non_max_suppression(model(img)[0], 0.25, 0.45, classes=None, agnostic=False)
        tb = PrettyTable(field_names=["Class", "Conf", "Point"])
        for res in pred:
            for *box, confidence, class_id in res:
                x1, y1, x2, y2 = map(int, box)
                label = f"{dataset_dict['names'][int(class_id)]}:{format(confidence, '.3f')}"
                tb.add_row(
                    [dataset_dict["names"][int(class_id)],
                     confidence, ",".join(map(lambda x: str(x), [x1, x2, y1, y2]))])
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
                cv2.rectangle(pred_img, (x1, y1), (x1 + (len(label) * 7), y1 - 20), (0, 0, 255), thickness=-1)
                cv2.putText(pred_img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        print(tb)
        cv2.imwrite(f"{save_dir}\\{name}", pred_img)
        pred_queue.put(pred_img)
    except:
        pass


def display_img():
    while True:
        try:
            img = pred_queue.get(timeout=10)
            cv2.imshow('Image', img)
            cv2.waitKey(0)
        except Exception as e:
            break


if __name__ == "__main__":
    mp.freeze_support()
    for i in list(os.walk(img_dir))[0][2]:
        Thread(target=read_file_pretreatment, args=(f"{os.path.abspath(img_dir)}\\{i}", i,)).start()
    display_img()
