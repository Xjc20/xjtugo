"""
文件功能: 随机围棋玩家，用于测试和基准对比

主要功能:
1. RandomPlayer类: 从所有合法落子位置中随机选择
2. 合法性检查: 使用GO类的valid_place_check方法验证落子
3. 默认PASS: 当没有合法落子位置时返回"PASS"

参数说明:
- go: GO类实例，包含当前游戏状态
- piece_type: 棋子类型 (1=黑子X, 2=白子O)
- possible_placements: 合法落子位置列表

使用方法:
1. 命令行运行: python random_player.py
   (从input.txt读取输入，输出到output.txt)
2. 作为模块导入: from random_player import RandomPlayer

示例:
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = RandomPlayer()
    action = player.get_input(go, piece_type)
    writeOutput(action)

特点:
- 简单易实现，适合作为AI测试的对手
- 完全随机决策，无策略性
- 可用于验证游戏逻辑的正确性
"""

import random
import sys
from read import readInput
from write import writeOutput

from host import GO

class RandomPlayer():
    """
    随机围棋玩家类
    
    该类实现了一个最简单的围棋AI，从所有合法落子位置中随机选择。
    主要用于:
    1. 作为基准测试对手，验证其他AI的性能
    2. 测试围棋游戏逻辑的正确性
    3. 提供最简单的AI实现示例
    
    属性:
        type: 玩家类型标识，固定为'random'
    """
    
    def __init__(self):
        """
        初始化随机玩家
        
        设置玩家类型为'random'，用于标识这是一个随机决策的玩家
        """
        self.type = 'random'

    def get_input(self, go, piece_type):
        '''
        获取下一步落子位置
        
        遍历整个棋盘，找出所有合法落子位置，然后随机选择一个。
        如果没有合法落子位置，则返回"PASS"表示跳过回合。

        :param go: GO类实例，包含当前游戏状态（棋盘、规则等）
        :param piece_type: 棋子类型，1表示黑子('X')，2表示白子('O')
        :return: 
            - 元组 (row, column): 落子位置的行和列坐标（从0开始）
            - 字符串 "PASS": 表示没有合法位置，跳过回合
            
        实现逻辑:
            1. 初始化空列表possible_placements存储合法位置
            2. 双重循环遍历棋盘所有位置(i, j)
            3. 调用go.valid_place_check()检查每个位置是否可落子
               - test_check=True表示只检查，不实际落子
            4. 如果合法，将坐标(i, j)添加到列表
            5. 遍历完成后:
               - 如果列表为空，返回"PASS"
               - 否则使用random.choice()随机选择一个位置返回
        '''        
        # 存储所有合法落子位置的列表
        possible_placements = []
        
        # 遍历棋盘的每一行
        for i in range(go.size):
            # 遍历棋盘的每一列
            for j in range(go.size):
                # 检查在(i, j)位置落子是否合法
                # test_check=True表示仅验证，不修改棋盘状态
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    # 如果合法，添加到候选列表
                    possible_placements.append((i,j))

        # 判断是否有合法落子位置
        if not possible_placements:
            # 列表为空，没有可落子的位置
            # 返回"PASS"表示跳过本轮
            return "PASS"
        else:
            # 从所有合法位置中随机选择一个
            # random.choice()从列表中均匀随机选取一个元素
            return random.choice(possible_placements)

if __name__ == "__main__":
    """
    主程序入口
    
    当直接运行random_player.py时执行以下流程:
    1. 设置棋盘大小N=5（5x5围棋）
    2. 从input.txt读取当前游戏状态
    3. 创建GO实例并设置棋盘状态
    4. 创建RandomPlayer实例
    5. 调用get_input()获取落子决策
    6. 将决策结果写入output.txt
    
    文件交互:
        输入文件: input.txt
            第1行: piece_type (1或2)
            第2-6行: previous_board (上一步棋盘)
            第7-11行: board (当前棋盘)
        
        输出文件: output.txt
            - "x,y" 格式: 表示在(x,y)落子
            - "PASS": 表示跳过回合
    """
    
    # ============================================
    # 步骤1: 初始化参数
    # ============================================
    # 设置棋盘大小为5x5
    N = 5
    
    # ============================================
    # 步骤2: 读取输入
    # ============================================
    # 从input.txt读取游戏状态
    # piece_type: 当前玩家棋子类型 (1=黑子, 2=白子)
    # previous_board: 上一步的棋盘状态（用于KO规则判断）
    # board: 当前棋盘状态
    piece_type, previous_board, board = readInput(N)
    
    # ============================================
    # 步骤3: 创建游戏实例
    # ============================================
    # 创建GO类的实例，传入棋盘大小
    go = GO(N)
    # 设置棋盘状态，包括当前棋盘和上一步棋盘
    # 这一步会初始化游戏状态，包括计算被提掉的棋子等
    go.set_board(piece_type, previous_board, board)
    
    # ============================================
    # 步骤4: 创建玩家实例并获取决策
    # ============================================
    # 创建随机玩家实例
    player = RandomPlayer()
    # 调用get_input方法获取下一步落子位置
    # 传入游戏实例和当前玩家棋子类型
    action = player.get_input(go, piece_type)
    
    # ============================================
    # 步骤5: 输出结果
    # ============================================
    # 将决策结果写入output.txt
    # action可以是元组(row, col)或字符串"PASS"
    writeOutput(action)