"""
文件功能: 纯MCTS蒙特卡罗树搜索围棋AI玩家

主要功能:
1. MCTSPlayer类: 使用蒙特卡罗树搜索算法进行决策
2. Node类: MCTS树节点，记录访问次数和得分
3. TemporaryMove类: 临时落子上下文管理器，安全模拟棋盘状态
4. 综合评估函数: 考虑捕获、气的数量、连通性、势力范围、中心距离等

参数说明:
- piece_type: 棋子类型 (1=黑子, 2=白子)
- num_simulations: MCTS最大模拟次数 (默认1000)
- timeout: MCTS超时时间 (默认20秒)
- threshold: 提前终止阈值 (默认2.0)
- c: UCB1探索系数 (默认1.4)
- k: 每次扩展的子节点数 (默认3)

使用方法:
1. 命令行运行: python my_player_0309.py
   (从input.txt读取输入，输出到output.txt)
2. 作为模块导入: from my_player_0309 import MCTSPlayer

MCTS算法流程:
1. 选择(Selection): 使用UCB1公式选择子节点
2. 扩展(Expansion): 扩展前k个高潜力落子位置
3. 模拟(Simulation): 使用快速策略模拟对局
4. 反向传播(Backpropagation): 更新节点访问次数和得分

评估函数权重:
- 捕获棋子: 权重2
- 自身气的数量: 权重2
- 对手气减少: 权重2
- 中心距离: 权重-0.5
- 连通性: 权重1
- 势力范围: 权重1
"""

import random
import sys
import math
import time
from copy import deepcopy
from read import readInput
from write import writeOutput
from host import GO

# 临时落子类，用于临时改变棋盘状态并在操作结束后恢复
class TemporaryMove:
    def __init__(self, go, i, j, piece_type):
        # 传入的 GO 类实例，代表当前的围棋游戏状态
        self.go = go
        # 落子的行坐标
        self.i = i
        # 落子的列坐标
        self.j = j
        # 落子的棋子类型
        self.piece_type = piece_type
        # 保存当前棋盘的深拷贝，以便后续恢复
        self.original_board = deepcopy(go.board)
        # 保存当前被提掉的棋子列表，以便后续恢复
        self.original_died_pieces = go.died_pieces

    def __enter__(self):
        # 在指定位置落子
        self.go.board[self.i][self.j] = self.piece_type
        # 计算对手的棋子类型
        opponent_piece_type = 3 - self.piece_type
        # 移除对手被提掉的棋子
        self.go.died_pieces = self.go.remove_died_pieces(opponent_piece_type)
        return self.go

    def __exit__(self, exc_type, exc_value, traceback):
        # 恢复棋盘到原始状态
        self.go.board = self.original_board
        # 恢复被提掉的棋子列表到原始状态
        self.go.died_pieces = self.original_died_pieces

# 蒙特卡罗树搜索（MCTS）中的节点类
class Node:
    def __init__(self, parent, move, go):
        # 父节点
        self.parent = parent  
        # 导致到达此节点的落子位置，根节点为 None
        self.move = move      
        # 当前的游戏状态，是 GO 类的实例
        self.go = go          
        # 子节点字典，键为落子位置
        self.children = {}    
        # 该节点被访问的次数
        self.visits = 0       
        # 从模拟中累计得到的得分
        self.score = 0.0      

# 查找从 (i, j) 开始的同色相连棋子组
def get_group(go, i, j, piece_type):
    """
    查找从指定位置 (i, j) 开始的同色相连棋子组。

    :param go: GO 类的实例，代表当前的围棋游戏状态
    :param i: 起始位置的行坐标
    :param j: 起始位置的列坐标
    :param piece_type: 棋子类型，1 代表黑子，2 代表白子
    :return: 同色相连棋子组的位置列表
    """
    # 获取当前棋盘状态
    board = go.board
    # 如果起始位置的棋子颜色与指定颜色不同，返回空列表
    if board[i][j] != piece_type:
        return []
    # 已访问位置集合
    visited = set()
    # 相连棋子组列表
    group = []
    # 栈，用于深度优先搜索
    stack = [(i, j)]
    while stack:
        # 从栈中弹出一个位置
        x, y = stack.pop()
        if (x, y) not in visited:
            # 标记该位置为已访问
            visited.add((x, y))
            # 将该位置添加到相连棋子组列表中
            group.append((x, y))
            # 遍历该位置的四个相邻位置
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                # 检查相邻位置是否在棋盘内且棋子颜色与指定颜色相同
                if 0 <= nx < go.size and 0 <= ny < go.size:
                    if board[nx][ny] == piece_type:
                        # 将相邻位置添加到栈中
                        stack.append((nx, ny))
    return group

# 计算一组棋子的气的数量
def get_qi(go, group):
    """
    计算一组棋子的气的数量（即相邻的空点数量）。

    :param go: GO 类的实例，代表当前的围棋游戏状态
    :param group: 棋子组的位置列表
    :return: 该棋子组的气的数量
    """
    # 获取当前棋盘状态
    board = go.board
    # 气的集合
    qi = set()
    # 遍历棋子组中的每个棋子
    for x, y in group:
        # 遍历该棋子的四个相邻位置
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            # 检查相邻位置是否在棋盘内且为空点
            if 0 <= nx < go.size and 0 <= ny < go.size:
                if board[nx][ny] == 0:
                    # 将空点添加到气的集合中
                    qi.add((nx, ny))
    # 返回气的数量
    return len(qi)

# 获取给定棋子类型的所有合法落子位置，考虑禁着点规则
def get_valid_moves(go, piece_type):
    """
    获取给定棋子类型的所有合法落子位置，同时考虑禁着点规则。

    :param go: GO 类的实例，代表当前的围棋游戏状态
    :param piece_type: 棋子类型，1 代表黑子，2 代表白子
    :return: 合法落子位置的列表，每个位置用元组 (i, j) 表示
    """
    # 用于存储合法落子位置的列表
    possible_placements = []
    # 遍历棋盘的每一行
    for i in range(go.size):
        # 遍历棋盘的每一列
        for j in range(go.size):
            # 检查在 (i, j) 位置落子是否合法
            if go.valid_place_check(i, j, piece_type, test_check=True):
                # 使用临时落子上下文管理器
                with TemporaryMove(go, i, j, piece_type):
                    # 检查落子后是否会形成禁着点（与上一步棋盘状态相同）
                    if not go.compare_board(go.board, go.previous_board):
                        # 如果不是禁着点，将该位置添加到合法落子位置列表中
                        possible_placements.append((i, j))
    return possible_placements

# 从玩家的角度计算得分差异，考虑贴目规则
def get_score_difference(go, my_piece_type):
    """
    从玩家的角度计算得分差异，同时考虑贴目规则。

    :param go: GO 类的实例，代表当前的围棋游戏状态
    :param my_piece_type: 玩家的棋子类型，1 代表黑子，2 代表白子
    :return: 玩家得分与对手得分的差值
    """
    # 计算黑子的得分
    black_score = go.score(1)
    # 计算白子的得分
    white_score = go.score(2)
    # 如果玩家是黑子
    if my_piece_type == 1:
        # 黑子得分减去白子得分再减去贴目
        return black_score - white_score - go.komi
    else:
        # 白子得分加上贴目再减去黑子得分
        return white_score + go.komi - black_score

# 使用 UCB1 公式选择一个子节点
def select_child(node):
    """
    使用 UCB1 公式选择一个子节点。

    :param node: 当前节点
    :return: 被选中的子节点
    """
    # 初始化最佳子节点为 None
    best_child = None
    # 初始化最佳得分
    best_score = -float('inf')
    # UCB1 公式中的探索系数，调整为 1.4
    c = 1.4
    # 遍历当前节点的所有子节点
    for child in node.children.values():
        # 如果子节点还未被访问过，优先选择该子节点
        if child.visits == 0:
            return child
        # 计算 UCB1 得分
        ucb_score = child.score / child.visits + c * math.sqrt(math.log(node.visits + 1e-10) / (child.visits + 1e-10))
        # 如果当前子节点的 UCB1 得分高于最佳得分
        if ucb_score > best_score:
            # 更新最佳得分
            best_score = ucb_score
            # 更新最佳子节点
            best_child = child
    return best_child

# 检查在指定位置落子能立即捕获的对手棋子数量
def immediate_capture(go, move, piece_type):
    """
    检查在指定位置落子能立即捕获的对手棋子数量。

    :param go: GO 类的实例，代表当前的围棋游戏状态
    :param move: 落子位置，用元组 (i, j) 表示
    :param piece_type: 落子的棋子类型，1 代表黑子，2 代表白子
    :return: 能立即捕获的对手棋子数量
    """
    # 使用临时落子上下文管理器
    with TemporaryMove(go, move[0], move[1], piece_type):
        # 计算对手的棋子类型
        opponent_piece_type = 3 - piece_type
        # 获取被提掉的对手棋子列表
        captured = go.died_pieces
        # 返回被提掉的对手棋子数量
        return len(captured)

# 计算落子位置到棋盘中心的距离
def distance_to_center(move, size):
    """
    计算落子位置到棋盘中心的距离。

    :param move: 落子位置，用元组 (i, j) 表示
    :param size: 棋盘的大小
    :return: 落子位置到棋盘中心的曼哈顿距离
    """
    # 计算棋盘中心的坐标
    center_x, center_y = size // 2, size // 2
    # 解包落子位置的坐标
    x, y = move
    # 计算曼哈顿距离
    return abs(x - center_x) + abs(y - center_y)

# 计算势力范围
def calculate_territory(go, piece_type):
    """
    计算指定棋子类型的势力范围。

    :param go: GO 类的实例，代表当前的围棋游戏状态
    :param piece_type: 棋子类型，1 代表黑子，2 代表白子
    :return: 势力范围的大小
    """
    territory = 0
    visited = set()
    for i in range(go.size):
        for j in range(go.size):
            if go.board[i][j] == 0 and (i, j) not in visited:
                group = get_group(go, i, j, 0)
                for pos in group:
                    visited.add(pos)
                surrounded_by = set()
                for x, y in group:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < go.size and 0 <= ny < go.size:
                            if go.board[nx][ny] != 0:
                                surrounded_by.add(go.board[nx][ny])
                if len(surrounded_by) == 1 and piece_type in surrounded_by:
                    territory += len(group)
    return territory

# 综合评估落子位置的价值
def evaluate_move(go, move, piece_type):
    """
    综合评估落子位置的价值，考虑立即捕获棋子数量、到棋盘中心的距离、自身棋子气的数量、对手棋子气的减少程度、棋子的连通性和势力范围。

    :param go: GO 类的实例，代表当前的围棋游戏状态
    :param move: 落子位置，用元组 (i, j) 表示
    :param piece_type: 落子的棋子类型，1 代表黑子，2 代表白子
    :return: 落子位置的综合评估得分
    """
    # 立即捕获的棋子数量
    capture_count = immediate_capture(go, move, piece_type)
    # 到棋盘中心的距离
    dist = distance_to_center(move, go.size)

    # 计算落子前对手所有棋子组的总气数
    opponent_piece_type = 3 - piece_type
    opponent_total_qi_before = 0
    visited = set()
    for i in range(go.size):
        for j in range(go.size):
            if go.board[i][j] == opponent_piece_type and (i, j) not in visited:
                group = get_group(go, i, j, opponent_piece_type)
                for pos in group:
                    visited.add(pos)
                opponent_total_qi_before += get_qi(go, group)

    # 落子后自身棋子的气的数量
    with TemporaryMove(go, move[0], move[1], piece_type):
        group = get_group(go, move[0], move[1], piece_type)
        qi = get_qi(go, group)

        # 计算落子后对手所有棋子组的总气数
        opponent_total_qi_after = 0
        visited = set()
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == opponent_piece_type and (i, j) not in visited:
                    group = get_group(go, i, j, opponent_piece_type)
                    for pos in group:
                        visited.add(pos)
                    opponent_total_qi_after += get_qi(go, group)

        # 计算棋子的连通性，这里用相连棋子的数量表示
        connectivity = len(group)

        # 计算落子后的势力范围
        territory = calculate_territory(go, piece_type)

    # 对手棋子气的减少程度
    opponent_qi_reduction = opponent_total_qi_before - opponent_total_qi_after

    # 综合评估得分，捕获棋子数量权重为 2，自身气的数量权重为 2，对手气的减少程度权重为 2，到中心距离权重为 -0.5，连通性权重为 1，势力范围权重为 1
    score = 2 * capture_count + 2 * qi + 2 * opponent_qi_reduction - 0.5 * dist + 1 * connectivity + 1 * territory
    return score

# 使用快速策略模拟一局游戏
def simulate_fast(go, my_piece_type):
    """
    使用快速策略模拟一局游戏。

    :param go: GO 类的实例，代表当前的围棋游戏状态
    :param my_piece_type: 玩家的棋子类型，1 代表黑子，2 代表白子
    :return: 模拟结束后玩家的得分与对手得分的差值
    """
    # 复制当前棋盘状态
    current_go = go.copy_board()
    # 增加模拟步数到 20
    for _ in range(20):
        # 当前落子的棋子类型
        current_piece_type = 1 if current_go.X_move else 2
        # 获取当前棋子类型的所有合法落子位置
        moves = get_valid_moves(current_go, current_piece_type)
        # 如果没有合法落子位置，结束模拟
        if not moves:
            break
        # 优先选择综合评估得分最高的位置落子
        best_move = max(moves, key=lambda m: evaluate_move(current_go, m, current_piece_type))
        # 在最佳位置落子
        current_go.place_chess(best_move[0], best_move[1], current_piece_type)
        # 移除被提掉的对手棋子
        current_go.died_pieces = current_go.remove_died_pieces(3 - current_piece_type)
        # 增加步数计数
        current_go.n_move += 1
        # 切换落子方
        current_go.X_move = not current_go.X_move
    # 计算模拟结束后玩家的得分与对手得分的差值
    score = get_score_difference(current_go, my_piece_type)
    # print(f"[MCTS] Simulation completed: score={score:.4f}", file=sys.stderr)
    return score

# 从给定节点扩展前 k 个高潜力的落子位置
def expand_node(node, piece_type, k=3):
    """
    从给定节点扩展前 k 个高潜力的落子位置。

    :param node: 当前节点
    :param piece_type: 落子的棋子类型，1 代表黑子，2 代表白子
    :param k: 要扩展的高潜力落子位置的数量，默认为 3
    """
    # 获取当前节点对应棋盘状态下该棋子类型的所有合法落子位置
    possible_moves = get_valid_moves(node.go, piece_type)
    # 根据综合评估得分对合法落子位置进行降序排序
    sorted_moves = sorted(possible_moves, key=lambda m: evaluate_move(node.go, m, piece_type), reverse=True)
    # 选取前 k 个高潜力的落子位置
    top_k_moves = sorted_moves[:k]

    # 遍历前 k 个高潜力的落子位置
    for move in top_k_moves:
        # 复制当前节点的棋盘状态
        new_go = node.go.copy_board()
        # 在新的棋盘状态下的指定位置落子
        new_go.place_chess(move[0], move[1], piece_type)
        # 移除被提掉的对手棋子
        new_go.died_pieces = new_go.remove_died_pieces(3 - piece_type)
        # 增加步数计数
        new_go.n_move += 1
        # 切换落子方
        new_go.X_move = not new_go.X_move
        # 创建新的子节点并添加到当前节点的子节点字典中
        node.children[move] = Node(node, move, new_go)
    print(f"[MCTS] Expansion completed: expanded {len(top_k_moves)} child nodes", file=sys.stderr)

# 运行蒙特卡罗树搜索（MCTS）来寻找最佳落子位置
def mcts(go, piece_type, num_simulations=400, timeout=20, threshold=2.0):
    """
    运行蒙特卡罗树搜索（MCTS）来寻找最佳落子位置，同时设置模拟次数上限和超时时间。

    :param go: GO 类的实例，代表当前的围棋游戏状态
    :param piece_type: 玩家的棋子类型，1 代表黑子，2 代表白子
    :param num_simulations: 最大模拟次数，默认为 1000
    :param timeout: 超时时间（秒），默认为 10 秒
    :param threshold: 判断是否有特别好的走法的阈值，默认为 2.0
    :return: 最佳落子位置，用元组 (i, j) 表示
    """
    # 创建根节点
    root = Node(None, None, go)
    # 扩展根节点的前 k 个高潜力落子位置
    expand_node(root, piece_type)
    print(f"[MCTS] Root expansion done, starting simulations...", file=sys.stderr)

    # 记录开始时间
    start_time = time.time()
    # 模拟次数计数器
    sim_count = 0
    # 最佳落子位置
    best_move = None
    # 最佳得分
    best_score = -float('inf')

    # 在未达到最大模拟次数且未超时的情况下进行模拟
    while sim_count < num_simulations and (time.time() - start_time) < timeout:
        # 从根节点开始
        node = root
        # 选择子节点直到到达叶子节点
        while node.children:
            node = select_child(node)
        # 对叶子节点对应的棋盘状态进行快速模拟
        score = simulate_fast(node.go, piece_type)
        # 反向传播得分
        while node:
            node.visits += 1
            node.score += score
            node = node.parent

        # 存储所有子节点的平均得分
        scores = []
        # 遍历根节点的所有子节点
        for move, child in root.children.items():
            # 计算子节点的平均得分
            avg_score = child.score / child.visits if child.visits > 0 else 0
            # 将平均得分添加到得分列表中
            scores.append(avg_score)
            # 如果当前平均得分高于最佳得分，更新最佳得分和最佳落子位置
            if avg_score > best_score:
                best_score = avg_score
                best_move = move

        # 如果有多个子节点的得分
        if len(scores) > 1:
            # 对得分列表进行降序排序
            sorted_scores = sorted(scores, reverse=True)
            # 如果得分最高的子节点与得分第二高的子节点的得分差值大于阈值
            if sorted_scores[0] - sorted_scores[1] > threshold:
                # 提前结束搜索
                print(f"[MCTS] Early termination at simulation {sim_count + 1}: best-second > {threshold}", file=sys.stderr)
                break

        # 增加模拟次数计数器
        sim_count += 1

    print(f"[MCTS] Search completed: {sim_count} simulations, best_move={best_move}, best_score={best_score:.4f}", file=sys.stderr)
    return best_move

# MCTS 玩家类
class MCTSPlayer:
    def __init__(self):
        # 玩家类型为 MCTS
        self.type = 'mcts'

    def get_input(self, go, piece_type):
        """
        使用 MCTS 启发式策略获取一个落子位置。

        :param go: GO 类的实例，代表当前的围棋游戏状态
        :param piece_type: 玩家的棋子类型，1 代表黑子，2 代表白子
        :return: 落子位置，用元组 (i, j) 表示；如果没有合法落子位置，返回 "PASS"
        """
        # 获取当前棋子类型的所有合法落子位置
        possible_moves = get_valid_moves(go, piece_type)
        # 如果没有合法落子位置
        if not possible_moves:
            # 返回 "PASS"
            return "PASS"
        else:
            # 运行 MCTS 算法寻找最佳落子位置
            best_move = mcts(go.copy_board(), piece_type)
            return best_move

if __name__ == "__main__":
    # 棋盘大小
    N = 5
    # 读取输入的棋子类型、上一步棋盘状态和当前棋盘状态
    piece_type, previous_board, board = readInput(N)
    # 创建 GO 类的实例
    go = GO(N)
    # 设置棋盘状态
    go.set_board(piece_type, previous_board, board)
    # 创建 MCTS 玩家实例
    player = MCTSPlayer()
    # 获取玩家的落子位置
    action = player.get_input(go, piece_type)
    # 输出落子位置
    writeOutput(action)