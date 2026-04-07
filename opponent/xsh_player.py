"""
文件功能: Alpha-Beta剪枝围棋AI玩家，使用Minimax算法和启发式评估函数

主要功能:
1. my_player类: 使用Alpha-Beta剪枝的Minimax算法进行决策
2. 启发式评估函数: 综合考虑中心控制、棋子连接性、气数、眼数、干扰对手等因素
3. 位置优化: 只考虑已有棋子周围的空位，减少搜索空间
4. 开局策略: 优先占据棋盘中心(2,2)

参数说明:
- piece_type: 棋子类型 (1=黑子, 2=白子)
- size: 棋盘大小 (默认5)
- max_depth: Minimax搜索深度 (默认2)
- alpha/beta: Alpha-Beta剪枝边界值
- center_weights: 中心控制权重矩阵

评估函数组成:
1. 中心控制: 越靠近中心得分越高 (权重20)
2. 棋子连接性: 相邻同色棋子加分 (权重5)
3. 气数潜力: 总气数得分 (权重50)
4. 干扰对手: 减少对手气数 (权重-10)
5. 眼数: 每个眼加300分
6. 棋子数量差: (我方-对方)*50
7. 低气惩罚: 气数<=2时减50分

使用方法:
1. 命令行运行: python xsh_player.py
   (从input.txt读取输入，输出到output.txt)
2. 作为模块导入: from xsh_player import my_player

算法特点:
- 使用deque进行BFS搜索
- 缓存优化避免重复计算
- Alpha-Beta剪枝减少搜索节点
"""

import random
import sys
from read import readInput
from write import writeOutput
from collections import deque 
from functools import lru_cache
import numpy as np
from host import GO
from copy import deepcopy

class my_player():
    def __init__(self):
        self.type = 'AI'
        self.size = 5
        self.possible_placements = []

    def get_possible_placements(self, go: GO, piece_type):
        self.possible_placements = []
        occupied = set()  # 记录已有棋子的位置

        for i in range(self.size):
            for j in range(self.size):
                if go.board[i][j] != 0:
                    occupied.add((i, j))

        # 只考虑已落子周围的空位
        for (i, j) in occupied:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < self.size and 0 <= nj < self.size and go.board[ni][nj] == 0:
                    if go.valid_place_check(ni, nj, piece_type, test_check=True):
                        self.possible_placements.append((ni, nj))

        return self.possible_placements

    def count_liberties(self, go: GO, pieces: list[tuple[int, int]]) -> int:
        """
        计算棋盘上指定类型的所有棋子的总气数。
        :param go: GO 实例
        :param piece_type: 棋子类型（1 或 2）
        :param pieces: 该类棋子的所有坐标 [(x1, y1), (x2, y2), ...]
        :return: 所有棋子的总气数
        """
        visited = set()  # 记录已经访问过的棋子
        total_liberties = 0

        for x, y in pieces:
            if (x, y) not in visited:
                # 计算当前棋块的气数
                liberties, block = self._count_block_liberties(go, x, y)
                total_liberties += liberties
                visited.update(block)  # 标记整个棋块为已访问

        return total_liberties

    def _count_block_liberties(self, go: GO, x: int, y: int) -> tuple[int, set]:
        """
        计算一个棋块的气数，并返回该棋块的所有坐标。
        :param go: GO 实例
        :param x: 棋块中的某个棋子的横坐标
        :param y: 棋块中的某个棋子的纵坐标
        :return: (气数, 棋块的所有坐标)
        """
        visited = set()
        queue = deque([(x, y)])
        liberties = 0
        piece_type = go.board[x][y]

        while queue:
            px, py = queue.popleft()
            if (px, py) in visited:
                continue
            visited.add((px, py))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < go.size and 0 <= ny < go.size:
                    if go.board[nx][ny] == 0:  # 空位
                        liberties += 1
                    elif go.board[nx][ny] == piece_type and (nx, ny) not in visited:
                        queue.append((nx, ny))

        return liberties, visited

    def is_surrounded_by(self, go, x, y, piece_type):
        """
        判断一个空点是否被己方棋子完全包围。
        :param go: GO 实例
        :param x: 空点的横坐标
        :param y: 空点的纵坐标
        :param piece_type: 棋子类型（1 或 2）
        :return: 如果空点被己方棋子完全包围，返回 True；否则返回 False
        """
        cnt = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < go.size and 0 <= ny < go.size:
                if go.board[nx][ny] != piece_type:  # 如果周围有非己方棋子或空位
                    cnt += 1
                    if cnt >= 2:
                        return False
        return True

    def count_eyes(self, go, x, y, piece_type):
        """
        计算某块棋子的眼的数量。
        :param go: GO 实例
        :param x: 棋块中的某个棋子的横坐标
        :param y: 棋块中的某个棋子的纵坐标
        :param piece_type: 棋子类型（1 或 2）
        :return: 该棋块的眼的数量
        """
        visited = set()
        queue = deque([(x, y)])
        eyes = 0

        while queue:
            px, py = queue.popleft()
            if (px, py) in visited:
                continue
            visited.add((px, py))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < go.size and 0 <= ny < go.size:
                    if go.board[nx][ny] == 0:  # 空位
                        # 检查该空位是否被己方棋子完全包围
                        if self.is_surrounded_by(go, nx, ny, piece_type):
                            eyes += 1
                    elif go.board[nx][ny] == piece_type and (nx, ny) not in visited:
                        queue.append((nx, ny))
        return eyes
    
    def count_all_eyes(self, go: GO, piece_type: int, pieces: list[tuple[int, int]]) -> int:
        """
        计算棋盘上指定类型的所有棋子的眼的总数量。
        :param go: GO 实例
        :param piece_type: 棋子类型（1 或 2）
        :param pieces: 该类棋子的所有坐标 [(x1, y1), (x2, y2), ...]
        :return: 所有棋子的眼的总数量
        """
        visited = set()  # 记录已经访问过的棋子
        total_eyes = 0

        for x, y in pieces:
            if (x, y) not in visited:
                # 计算当前棋块的眼的数量
                eyes = self.count_eyes(go, x, y, piece_type)
                total_eyes += eyes
                # 标记整个棋块为已访问
                block = self._get_block(go, x, y, piece_type)
                visited.update(block)

        return total_eyes

    def _get_block(self, go: GO, x: int, y: int, piece_type: int) -> set:
        """
        获取与指定棋子相连的整个棋块的所有坐标。
        :param go: GO 实例
        :param x: 棋块中的某个棋子的横坐标
        :param y: 棋块中的某个棋子的纵坐标
        :param piece_type: 棋子类型（1 或 2）
        :return: 棋块的所有坐标
        """
        visited = set()
        queue = deque([(x, y)])

        while queue:
            px, py = queue.popleft()
            if (px, py) in visited:
                continue
            visited.add((px, py))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < go.size and 0 <= ny < go.size:
                    if go.board[nx][ny] == piece_type and (nx, ny) not in visited:
                        queue.append((nx, ny))

        return visited
    
    def evaluate_connectivity(self, go: GO, piece_type: int, pieces: list[tuple[int, int]]) -> int:
        """
        评估棋盘上指定类型棋子的连接性。
        :param go: GO 实例
        :param piece_type: 棋子类型（1 或 2）
        :param pieces: 该类棋子的所有坐标 [(x1, y1), (x2, y2), ...]
        :return: 连接性得分
        """
        score = 0
        visited = set()  # 避免重复计算

        for x, y in pieces:
            if (x, y) not in visited:
                # 遍历四个方向
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < go.size and 0 <= ny < go.size:
                        if go.board[nx][ny] == piece_type:
                            score += 3  # 直接连接加 3 分
                visited.add((x, y))  # 标记已访问节点

        return score

    def get_my_pieces(self, go: GO, piece_type: int) -> list:
        """
        获取棋盘上我方已落子的所有坐标。
        :param go: GO 实例
        :param piece_type: 我方棋子类型（1 或 2）
        :return: 包含我方棋子坐标的列表 [(x1, y1), (x2, y2), ...]
        """
        my_pieces = []
        for x in range(go.size):
            for y in range(go.size):
                if go.board[x][y] == piece_type:  # 如果该位置是我方棋子
                    my_pieces.append((x, y))
        return my_pieces

    def calculate_score(self, go:GO, piece_type):
        score = 0
        my_pieces = self.get_my_pieces(go, piece_type)
        opponent_pieces = self.get_my_pieces(go, 3 - piece_type)

        # 1. 中心控制权重（越靠近中心分越高）
        center_weights = np.array([
            [0.2, 0.8, 1.0, 0.8, 0.2],
            [0.8, 1.0, 1.2, 1.2, 0.8],
            [1.0, 1.2, 1.5, 1.0, 1.0],
            [0.8, 1.0, 1.2, 1.0, 0.8],
            [0.2, 0.8, 1.0, 0.8, 0.2]
        ])
        
        for x, y in my_pieces:
            score += center_weights[x][y] * 20

            # 如果气数小于等于 2，给予惩罚
            liberties, _ = self._count_block_liberties(go, x, y)
            if liberties <= 2:  
                score -= 50


        # 2. 棋子连接性（相邻同色棋子数量）
        connectivity_score = self.evaluate_connectivity(go, piece_type, my_pieces)
        score += connectivity_score * 5  # 增加权重

        # 3. 气数潜力
        liberties = self.count_liberties(go, my_pieces)
        score += liberties * 50
        
        # 4. 干扰对手（靠近敌方棋子）
        opponent_liberties = self.count_liberties(go, opponent_pieces)
        score -= opponent_liberties * 10

        # 5. 尽可能做出眼
        eyes = self.count_all_eyes(go, piece_type, my_pieces)
        score += 300 * eyes

        # 6. 计算双方棋子数目
        my_pieces_count = len(my_pieces)
        opponent_pieces_count = len(opponent_pieces)
        score += (my_pieces_count - opponent_pieces_count) * 50
        
        return score
    
    
    max_depth = 2
    # Alpha-Beta 剪枝
    def cached_minimax(self, piece_type, go, depth, alpha, beta, is_maximizing):

        if depth >= self.max_depth or go.game_end(piece_type):
            return self.calculate_score(go, piece_type)   
        
        if is_maximizing: #最大最小化
            
            best_score = -float('inf') #初始化best_score
            for move in self.get_possible_placements(go, piece_type):
                x, y = move  # 将元组拆分为两个整数
                go.place_chess(x, y, piece_type)  # AI 下棋
                died_pieces = go.remove_died_pieces(3-piece_type)  # 移除对方的死子
                score = self.cached_minimax(3-piece_type, go, depth+1, alpha, beta, False)#递归调用，轮换至对方下棋
                # print(f"alpha: {alpha} score: {score}")
                go.board[x][y] = 0  # 还原棋盘
                for (i, j) in died_pieces:
                    go.board[i][j] = 3 - piece_type
                best_score = max(score, best_score) #更新最佳得分  
                alpha = max(alpha, score)
                if beta <= alpha:  # Beta 剪枝
                    break
            return best_score
        else:
            best_score = float('inf')
            for move in self.get_possible_placements(go, piece_type):
                x, y = move
                go.place_chess(x, y, piece_type)  # AI 下棋
                died_pieces = go.remove_died_pieces(3-piece_type)  # 移除对方的死子
                score = self.cached_minimax(3-piece_type, go, depth+1, alpha, beta, True)
                # print(f"beta: {beta} score: {score}")
                go.board[x][y] = 0  # 还原棋盘
                for (i, j) in died_pieces:
                    go.board[i][j] = 3 - piece_type
                best_score = min(score, best_score)
                beta = min(beta, score)
                if beta <= alpha:  # Alpha 剪枝
                    break
            return best_score
    
    def minimax_alpha_beta(self, piece_type, go:GO, depth, alpha, beta, is_maximizing):
        # rotation, is_mirror = self.board_normalize(go)
        # 检查缓存
        cached_result = self.cached_minimax(piece_type, go, depth, alpha, beta, is_maximizing)
        return cached_result
    
    #AI走子
    def get_input(self, go:GO, piece_type):
        if go.board[2][2] == 0:
            return 2,2 
        
        possible_placement = self.get_possible_placements(go, piece_type)

        if len(possible_placement) == 1:  # r只有 1 个合法位置，直接下
            return possible_placement[0]
        if not possible_placement:
            return "PASS"
        
        best_score = -float('inf')
        best_move = None
        alpha, beta = -float('inf'), float('inf')
        
        for move in possible_placement:#遍历合法走子
            x, y = move
            go.place_chess(x, y, piece_type)  # AI 下棋
            died_pieces = go.remove_died_pieces(3-piece_type)  # 移除对方的死子
            score = self.minimax_alpha_beta(piece_type, go, 0, alpha, beta, False)
            go.board[x][y] = 0  # 还原棋盘
            for (i, j) in died_pieces:
                go.board[i][j] = 3 - piece_type
            if score > best_score:#更新得分和走法
                best_score = score
                best_move = move
            # print(f"score: {score} best_score: {best_score} best_move: {best_move}" )
            # best_move = self.reverse_transform(best_move[0], best_move[1], self.size, rotation, is_mirror)
        
    # 如果没有找到最佳走法，随机选择一个合法位置
        if best_move == None:
            best_move = random.choice(possible_placement)
        
        return best_move

if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = my_player()
    action = player.get_input(go, piece_type)
    writeOutput(action)