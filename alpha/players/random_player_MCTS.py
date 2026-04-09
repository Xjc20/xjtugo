"""
MCTS（蒙特卡罗树搜索）围棋AI玩家

文件功能:
    基于蒙特卡罗树搜索(Monte Carlo Tree Search)算法实现的围棋AI玩家
    
主要功能:
    1. MCTSPlayer类: 使用MCTS算法进行决策
    2. Node类: MCTS树节点，记录访问次数和得分
    3. 综合评估函数: 考虑气的数量、提吃、位置价值等
    
MCTS算法流程:
    1. 选择(Selection): 使用UCB1公式选择最优子节点
    2. 扩展(Expansion): 随机选择一个未探索的落子位置扩展树
    3. 模拟(Simulation): 使用快速策略模拟对局到一定深度
    4. 反向传播(Backpropagation): 更新路径上所有节点的访问次数和得分
    
参数说明:
    - piece_type: 棋子类型 (1=黑子X, 2=白子O)
    - num_simulations: MCTS最大模拟次数 (默认2000)
    - timeout: MCTS超时时间 (默认9.5秒)
    - c: UCB1探索系数 (默认1.4)
    
使用方法:
    python my_player.py
    (从input.txt读取输入，输出到output.txt)
"""

import random
import sys
import math
import time
from copy import deepcopy
from read import readInput
from write import writeOutput
from host import GO


class Node:
    """
    MCTS树节点类
    
    表示蒙特卡罗树搜索中的一个节点，代表一个特定的棋盘状态。
    每个节点记录:
        - 从父节点到达此节点的落子位置
        - 当前游戏状态
        - 子节点列表
        - 访问次数
        - 累计得分
        - 未尝试的落子位置
    
    属性:
        parent: 父节点引用
        move: 导致到达此节点的落子位置 (row, col)
        go: 当前游戏状态 (GO类实例)
        piece_type: 当前玩家棋子类型
        children: 子节点字典，键为落子位置
        visits: 该节点被访问的次数
        score: 从模拟中累计得到的得分
        untried_moves: 尚未尝试的落子位置列表
    """
    
    def __init__(self, parent, move, go, piece_type):
        """
        初始化MCTS节点
        
        参数:
            parent: 父节点
            move: 导致到达此节点的落子位置
            go: GO类实例，表示当前游戏状态
            piece_type: 当前玩家棋子类型
        """
        self.parent = parent
        self.move = move
        self.go = go
        self.piece_type = piece_type
        self.children = {}
        self.visits = 0
        self.score = 0.0
        self.untried_moves = None

    def get_untried_moves(self):
        """
        获取尚未尝试的合法落子位置
        
        如果未初始化，则获取当前节点状态下的所有合法落子位置
        
        返回:
            list: 合法落子位置列表，每个元素为(row, col)元组
        """
        if self.untried_moves is None:
            self.untried_moves = get_valid_moves(self.go, self.piece_type)
        return self.untried_moves

    def is_fully_expanded(self):
        """
        检查节点是否完全扩展
        
        当所有合法落子位置都被尝试过（成为子节点）时，节点完全扩展
        
        返回:
            bool: True表示完全扩展，False表示还有未尝试的落子
        """
        return len(self.get_untried_moves()) == 0

    def best_child(self, c=1.4):
        """
        使用UCB1公式选择最优子节点
        
        UCB1 (Upper Confidence Bound 1) 公式:
            UCB = 平均得分 + c * sqrt(ln(父节点访问次数) / 子节点访问次数)
        
        这个公式平衡了:
            - 利用(exploitation): 选择得分高的节点
            - 探索(exploration): 优先选择访问次数少的节点
        
        参数:
            c: 探索系数，控制探索与利用的平衡
            
        返回:
            Node: UCB值最大的子节点
        """
        best_score = -float('inf')
        best_node = None
        for child in self.children.values():
            if child.visits == 0:
                ucb_score = float('inf')
            else:
                ucb_score = child.score / child.visits + c * math.sqrt(math.log(self.visits) / child.visits)
            if ucb_score > best_score:
                best_score = ucb_score
                best_node = child
        return best_node


def get_group(go, i, j, piece_type):
    """
    查找从指定位置开始的同色相连棋子组（连通块）
    
    使用深度优先搜索(DFS)算法找出所有通过上下左右相连的同色棋子
    
    参数:
        go: GO类实例
        i: 起始行坐标
        j: 起始列坐标
        piece_type: 棋子类型 (1=黑子, 2=白子)
    
    返回:
        list: 同色相连棋子组的位置列表，每个元素为(row, col)元组
    """
    board = go.board
    if board[i][j] != piece_type:
        return []
    visited = set()
    group = []
    stack = [(i, j)]
    while stack:
        x, y = stack.pop()
        if (x, y) not in visited:
            visited.add((x, y))
            group.append((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < go.size and 0 <= ny < go.size:
                    if board[nx][ny] == piece_type:
                        stack.append((nx, ny))
    return group


def get_liberties(go, group):
    """
    计算一组棋子的气（liberties）的数量
    
    气是指棋子周围相邻的空位（可以落子的位置）
    
    参数:
        go: GO类实例
        group: 棋子组的位置列表
    
    返回:
        set: 气的位置集合（去重后的空位列表）
    """
    board = go.board
    liberties = set()
    for x, y in group:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < go.size and 0 <= ny < go.size:
                if board[nx][ny] == 0:
                    liberties.add((nx, ny))
    return liberties


def get_valid_moves(go, piece_type):
    """
    获取给定棋子类型的所有合法落子位置
    
    合法性检查包括:
    1. 位置为空（没有棋子）
    2. 落子后有气（不是自杀）
    3. 提吃对方棋子后有气
    4. 不违反KO规则（不能回到上一步的棋盘状态）
    
    参数:
        go: GO类实例，当前游戏状态
        piece_type: 棋子类型 (1=黑子, 2=白子)
    
    返回:
        list: 合法落子位置列表，每个元素为(row, col)元组
    """
    possible_placements = []
    for i in range(go.size):
        for j in range(go.size):
            if go.valid_place_check(i, j, piece_type, test_check=True):
                test_go = go.copy_board()
                test_go.board[i][j] = piece_type
                test_go.remove_died_pieces(3 - piece_type)
                if not test_go.compare_board(test_go.board, go.previous_board):
                    possible_placements.append((i, j))
    return possible_placements


def evaluate_position(go, move, piece_type):
    """
    评估落子位置的价值
    
    综合考虑多个因素:
    1. 气的数量 - 落子后自己有多少气
    2. 提吃奖励 - 是否能提吃对方的棋子
    3. 位置价值 - 中心>边>角落
    4. 距离中心的距离
    
    参数:
        go: GO类实例
        move: 落子位置 (row, col)
        piece_type: 棋子类型 (1=黑子, 2=白子)
    
    返回:
        float: 位置评估得分，越高越好
    """
    i, j = move
    opponent_type = 3 - piece_type

    board = go.board
    original_liberties = 0
    if board[i][j] == 0:
        test_go = go.copy_board()
        test_go.board[i][j] = piece_type
        test_go.remove_died_pieces(opponent_type)
        my_group = get_group(test_go, i, j, piece_type)
        if my_group:
            original_liberties = len(get_liberties(test_go, my_group))
    else:
        my_group = get_group(go, i, j, piece_type)
        if my_group:
            original_liberties = len(get_liberties(go, my_group))

    score = 0

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + dx, j + dy
        if 0 <= ni < go.size and 0 <= nj < go.size:
            if board[ni][nj] == opponent_type:
                opp_group = get_group(go, ni, nj, opponent_type)
                if opp_group:
                    opp_liberties = len(get_liberties(go, opp_group))
                    if opp_liberties == 1:
                        score += 10
                    elif opp_liberties == 2:
                        score += 3

    center = go.size // 2
    dist_to_center = abs(i - center) + abs(j - center)
    score -= dist_to_center * 0.5

    if i == 0 or i == go.size - 1 or j == 0 or j == go.size - 1:
        score -= 1
    elif i == 1 or i == go.size - 2 or j == 1 or j == go.size - 2:
        score += 0.5
    elif i == center and j == center:
        score += 2

    corner_positions = [(0, 0), (0, go.size-1), (go.size-1, 0), (go.size-1, go.size-1)]
    if (i, j) in corner_positions:
        score += 1

    return score + original_liberties * 2 + random.random() * 0.1


def make_move(go, i, j, piece_type):
    """
    在棋盘上执行落子操作（永久修改）
    
    参数:
        go: GO类实例
        i: 落子行坐标
        j: 落子列坐标
        piece_type: 棋子类型
    
    返回:
        list: 被提吃的对方棋子列表
    """
    go.previous_board = deepcopy(go.board)
    go.board[i][j] = piece_type
    died_pieces = go.remove_died_pieces(3 - piece_type)
    return died_pieces


def simulate(go, piece_type, depth=15):
    """
    模拟函数 - 从当前状态模拟到一定深度
    
    使用快速策略（基于评估函数）进行模拟，
    不进行完整的游戏，而是模拟一定数量的落子
    
    参数:
        go: GO类实例，初始游戏状态
        piece_type: 当前玩家棋子类型
        depth: 模拟的落子深度（默认15步）
    
    返回:
        float: 模拟结果得分（玩家得分 - 对手得分）
    """
    current_piece = piece_type
    sim_go = go.copy_board()
    moves_made = 0

    while moves_made < depth:
        valid_moves = get_valid_moves(sim_go, current_piece)

        if not valid_moves:
            opponent_valid = get_valid_moves(sim_go, 3 - current_piece)
            if not opponent_valid:
                break
            moves_made += 1
            current_piece = 3 - current_piece
            continue

        scored_moves = [(move, evaluate_position(sim_go, move, current_piece)) for move in valid_moves]
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        top_moves = [m[0] for m in scored_moves[:5]]

        if top_moves:
            weights = [10 - idx for idx in range(len(top_moves))]
            move = random.choices(top_moves, weights=weights, k=1)[0]
            make_move(sim_go, move[0], move[1], current_piece)

        moves_made += 1
        current_piece = 3 - current_piece

    black_score = sim_go.score(1)
    white_score = sim_go.score(2)

    if piece_type == 1:
        return black_score - white_score - sim_go.komi
    else:
        return white_score - black_score + sim_go.komi


class MCTSPlayer:
    """
    MCTS围棋玩家类
    
    使用蒙特卡罗树搜索(MCTS)算法进行决策的围棋AI玩家。
    
    属性:
        num_simulations: 最大模拟次数
        timeout: 超时时间（秒）
        c: UCB1探索系数
    """
    
    def __init__(self, num_simulations=2000, timeout=9.5, c=1.4):
        """
        初始化MCTS玩家
        
        参数:
            num_simulations: 最大模拟次数
            timeout: 超时时间（秒）
            c: UCB1探索系数，控制探索与利用的平衡
        """
        self.num_simulations = num_simulations
        self.timeout = timeout
        self.c = c

    def get_input(self, go, piece_type):
        """
        获取下一步落子位置
        
        MCTS算法主流程:
        1. 从根节点开始，重复以下步骤直到达到模拟次数或超时:
           a) 选择: 从根节点向下选择最优子节点
           b) 扩展: 添加新节点
           c) 模拟: 从新节点进行随机/启发式模拟
           d) 反向传播: 更新路径上所有节点的统计信息
        2. 返回访问次数最多的子节点对应的落子
        
        参数:
            go: GO类实例，当前游戏状态
            piece_type: 玩家棋子类型 (1=黑子, 2=白子)
        
        返回:
            tuple: 落子位置 (row, col)
            str: "PASS" 如果没有合法落子位置
        """
        start_time = time.time()
        root = Node(None, None, go.copy_board(), piece_type)

        simulations = 0
        while simulations < self.num_simulations:
            if time.time() - start_time > self.timeout:
                break

            node = root
            current_go = go.copy_board()
            current_piece = piece_type

            while node.children and not node.is_fully_expanded():
                if node.untried_moves is None:
                    node.untried_moves = get_valid_moves(node.go, node.piece_type)
                if node.untried_moves:
                    break
                node = node.best_child(self.c)
                if node is None:
                    break

                make_move(current_go, node.move[0], node.move[1], node.piece_type)
                current_piece = 3 - current_piece

            if node is None or (time.time() - start_time > self.timeout):
                break

            if node.untried_moves is None:
                node.untried_moves = get_valid_moves(node.go, node.piece_type)

            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)

                new_go = node.go.copy_board()
                make_move(new_go, move[0], move[1], current_piece)

                child_node = Node(node, move, new_go, 3 - current_piece)
                node.children[move] = child_node
                node = child_node
            else:
                if not node.children:
                    break
                node = node.best_child(self.c)
                if node is None:
                    break
                make_move(current_go, node.move[0], node.move[1], node.piece_type)
                current_piece = 3 - current_piece

            result = simulate(current_go, current_piece, depth=20)

            temp = node
            while temp is not None:
                temp.visits += 1
                temp.score += result
                temp = temp.parent

            simulations += 1

        if not root.children:
            valid_moves = get_valid_moves(go, piece_type)
            if not valid_moves:
                return "PASS"
            return max(valid_moves, key=lambda m: evaluate_position(go, m, piece_type))

        best_move = None
        best_score = -float('inf')

        for move, child in root.children.items():
            if child.visits > 0:
                avg_score = child.score / child.visits
                if avg_score > best_score:
                    best_score = avg_score
                    best_move = move

        if best_move is None:
            valid_moves = get_valid_moves(go, piece_type)
            if not valid_moves:
                return "PASS"
            return max(valid_moves, key=lambda m: evaluate_position(go, m, piece_type))

        return best_move


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)

    player = MCTSPlayer(num_simulations=200, timeout=20, c=2)
    action = player.get_input(go, piece_type)
    writeOutput(action)
