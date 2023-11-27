#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2023 - 2023 Gosu, Inc. All Rights Reserved 
#
# @Time    : 2023/11/24 17:39
# @Author  : GosuXX
# @File    : sql_manager.py.py

import functools

from sqlalchemy import create_engine, MetaData, Engine
from sqlalchemy.orm import Session, sessionmaker


class SqlException(Exception):
    """ xxx """

    def __init__(self, e):
        print(e)


class Db_manager:
    __instance = None
    __engine: Engine = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    @classmethod
    def db_session(cls, host=None, port=None, user=None, pwd=None, db=None):
        def method(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                e = Db_manager.gen_engine(host, port, user, pwd, db)
                s = Db_manager.session()
                m = Db_manager.meta()
                try:
                    res = func(session=s, meta=m, *args, **kwargs)
                    s.commit()
                    s.close()
                except Exception as e:
                    Db_manager.__engine.dispose()
                    raise SqlException(e)
                return res

            return wrapper

        return method

    @classmethod
    def gen_engine(cls, host=None, port=None, user=None, pwd=None, db=None) -> Engine:
        if (not Db_manager.__engine) or all(map(lambda x: x is not None, [host, port, user, pwd, db])):
            engine = create_engine(f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}",
                                   pool_pre_ping=True, max_overflow=10, pool_size=20, pool_recycle=600)
            Db_manager.__engine = engine
        return Db_manager.__engine

    @classmethod
    def session(cls) -> Session:
        s = sessionmaker(bind=Db_manager.__engine)
        return s()

    @classmethod
    def meta(cls) -> MetaData:
        m = MetaData()
        m.reflect(bind=Db_manager.__engine)
        return m
