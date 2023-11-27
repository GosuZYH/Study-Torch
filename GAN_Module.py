#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2023 - 2023 Gosu, Inc. All Rights Reserved
#
# @Time    : 2023/11/10 15:23
# @Author  : GosuXX
# @File    : FCN_Model.py

import torch.nn as nn
import torch.nn.functional as F


class FCN_Model(nn.Module):
    ''' 全连接网络 '''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
