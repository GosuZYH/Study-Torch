#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2023 - 2023 Gosu, Inc. All Rights Reserved 
#
# @Time    : 2023/12/8 15:37
# @Author  : GosuXX
# @File    : forTestImg.py
from time import sleep

import wx

# init
app = wx.App()
dc = wx.ScreenDC()

# set line and fill style
dc.SetBrush(wx.TRANSPARENT_BRUSH)
dc.SetPen(wx.Pen((0, 255, 0), width=10, style=wx.PENSTYLE_SOLID))

# draw (x, y, width, height)
dc.DrawRectangle(100, 100, 200, 100)
sleep(5000)
