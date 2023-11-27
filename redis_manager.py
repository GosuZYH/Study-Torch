#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2023 - 2023 Gosu, Inc. All Rights Reserved 
#
# @Time    : 2023/11/24 13:39
# @Author  : GosuXX
# @File    : redis_manager.py

import redis


r = redis.StrictRedis(host='localhost', port=6379, db=0)
r.set("zyh", 123)
res = r.get("zyh")
print(res)
