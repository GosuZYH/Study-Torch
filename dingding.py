#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2021 - 2021 Gosu, Inc. All Rights Reserved
#
# Time    : 2021/01/26 23:21
# Author  : GosuXX
# File    : dingding.py

import requests
import json
import socket

'''
钉钉webhook机器人 by zyh
'''


def robot_requests():
    # 获取运作电脑名称
    hostname = socket.gethostname()
    if not hostname:
        hostname = '暂未获取到'
    # 获取本机IP
    rpa_ip = socket.gethostbyname(hostname)
    if not rpa_ip:
        hostname = '暂未获取到'

    # 臭臭群机器人
    url1 = 'https://oapi.dingtalk.com/robot/send?access_token=1f09ef142f8aa069574855b188523cb732e86a72252f98a69b2d24900558aa5c'
    # ...更多群机器人
    url2 = ''

    # message = f"您询问的是 |T9音乐烤吧（江园路店）- 上海市 浦雪路269号109室| 吗，"
    message = f"小晴，你怎么和张宇航哥哥说话的！杀掉你喵~"
    # message = f"张宇航哥哥,这条消息发自Name:{hostname},Ip:{rpa_ip}的主机上"

    program = {
        "msgtype": "text",
        "text": {"content": f'{message}'},
    }

    headers = {'Content-Type': 'application/json'}
    f = requests.post(url1, data=json.dumps(program), headers=headers)


if __name__ == '__main__':
    robot_requests()
