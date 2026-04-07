"""
文件功能: 清理工具，用于清空比赛结果记录文件

主要功能:
1. 清空cresult.txt文件内容
2. 用于比赛开始前重置结果记录

参数说明:
- path: 文件路径前缀 (默认当前目录"./")
- cresult.txt: 比赛结果记录文件

使用方法:
1. 直接运行: python clean.py
2. 作为模块导入后调用

注意:
- 此操作会清空cresult.txt的所有内容
- 运行前请确保已保存需要的历史记录
"""

import numpy as numpy
import json

path = "./"

j = open(path + "cresult.txt", "w").close()

