"""
AlphaZero风格的MCTS搜索

包含:
- AlphaZeroNode: 带先验概率的MCTS节点
- AlphaZeroMCTS: PUCT搜索算法
- TemporaryMove: 安全的临时落子上下文管理器
- MCTSState: MCTS搜索状态管理器
"""

import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from host import GO
from copy import deepcopy
from alpha.network import encode_board


class TemporaryMove:
    """
    临时落子上下文管理器

    安全地模拟落子，操作结束后自动恢复棋盘状态
    """

    def __init__(self, go, i, j, piece_type):
        self.go = go
        self.i = i
        self.j = j
        self.piece_type = piece_type
        self.original_board = deepcopy(go.board)
        self.original_previous_board = deepcopy(go.previous_board)
        self.original_died_pieces = go.died_pieces

    def __enter__(self):
        self.go.previous_board = deepcopy(self.go.board)
        self.go.board[self.i][self.j] = self.piece_type
        opponent_piece_type = 3 - self.piece_type
        self.go.died_pieces = self.go.remove_died_pieces(opponent_piece_type)
        return self.go

    def __exit__(self, exc_type, exc_value, traceback):
        self.go.board = self.original_board
        self.go.previous_board = self.original_previous_board
        self.go.died_pieces = self.original_died_pieces


class MCTSState:
    """
    MCTS搜索状态管理器

    优化策略：
    1. 只深拷贝必要的 board 数组，不拷贝整个 GO 对象
    2. 支持 copy() 快速创建搜索路径上的状态副本
    3. 支持多次落子操作，exit 时恢复原始状态
    """

    def __init__(self, go, copy_board_array=True):
        if copy_board_array:
            self.board = deepcopy(go.board)
            self.previous_board = deepcopy(go.previous_board) if go.previous_board is not None else None
        else:
            self.board = go.board
            self.previous_board = go.previous_board
        self.size = go.size
        self.n_move = go.n_move
        self.X_move = go.X_move
        self.komi = go.komi
        self.verbose = go.verbose
        self.died_pieces = []
        self.captured_pieces = []

    def copy(self):
        """快速复制当前状态"""
        new_state = MCTSState.__new__(MCTSState)
        new_state.board = deepcopy(self.board)
        new_state.previous_board = deepcopy(self.previous_board) if self.previous_board is not None else None
        new_state.size = self.size
        new_state.n_move = self.n_move
        new_state.X_move = self.X_move
        new_state.komi = self.komi
        new_state.verbose = self.verbose
        new_state.died_pieces = []
        new_state.captured_pieces = []
        return new_state

    def place_chess(self, i, j, piece_type):
        """落子并处理提子"""
        self.previous_board = deepcopy(self.board)
        self.board[i][j] = piece_type
        opponent = 3 - piece_type
        died = []
        for di in range(self.size):
            for dj in range(self.size):
                if self.board[di][dj] == opponent:
                    group = self._get_group(di, dj, opponent)
                    if not self._has_liberty(group):
                        died.extend(group)
        for pos in died:
            self.board[pos[0]][pos[1]] = 0
        self.died_pieces = died
        self.n_move += 1
        self.X_move = not self.X_move

    def _get_group(self, i, j, piece_type):
        """获取连通块"""
        board = self.board
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
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if board[nx][ny] == piece_type:
                            stack.append((nx, ny))
        return group

    def _has_liberty(self, group):
        """检查连通块是否有气"""
        for x, y in group:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx][ny] == 0:
                        return True
        return False

    def score(self, piece_type):
        """获取棋子数量 (与 host.py 一致)"""
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == piece_type:
                    cnt += 1
        return cnt

    def _get_empty_group(self, i, j):
        """获取空白连通块"""
        board = self.board
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
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if board[nx][ny] == 0:
                            stack.append((nx, ny))
        return group


class AlphaZeroNode:
    """
    AlphaZero风格的MCTS节点

    融合了:
    - prior: 神经网络的先验概率
    - heuristic_score: 启发式评估得分 (来自0309)
    - PUCT选择公式
    """

    def __init__(self, parent=None, move=None, prior=0.0, state=None, to_play=None):
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.state = state
        self.to_play = to_play
    
    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def select_child(self, c_puct=1.5):
        """
        PUCT选择公式
        
        U(s,a) = c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))
        Q(s,a) = 平均价值
        Score = Q(s,a) + U(s,a)
        """
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            exploration = (c_puct * child.prior *
                          math.sqrt(self.visits + 1) / (1 + child.visits))
            score = -child.value() + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child


def get_valid_moves_mcts(state, piece_type, move_cache):
    """获取合法落子位置（缓存优化版）"""
    board_tuple = tuple(map(tuple, state.board))
    prev_tuple = tuple(map(tuple, state.previous_board)) if state.previous_board is not None else None
    cache_key = (board_tuple, prev_tuple, piece_type)
    if cache_key in move_cache:
        return move_cache[cache_key]

    valid_moves = []
    go = GO(state.size)
    go.set_board(piece_type, state.previous_board, state.board)
    for i in range(state.size):
        for j in range(state.size):
            if go.valid_place_check(i, j, piece_type, test_check=True):
                valid_moves.append((i, j))
    
    move_cache[cache_key] = valid_moves
    return valid_moves


class AlphaZeroMCTS:
    """
    AlphaZero风格的MCTS搜索
    
    融合优化:
    1. 神经网络先验概率 (PUCT)
    2. 启发式扩展: 只扩展前k个高潜力位置
    3. 提前终止: 最佳和次佳差距超过阈值时停止
    4. 快速模拟: 使用贪婪策略
    """
    
    def __init__(self, network, num_simulations=800, c_puct=1.5, device=None,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25, **_):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device if device is not None else next(network.parameters()).device
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.move_cache = {}
    
    def _action_to_index(self, move, size):
        if move == "PASS":
            return size * size
        return move[0] * size + move[1]
    
    def _is_terminal(self, node):
        if node.state.n_move >= node.state.size * node.state.size - 1:
            return True
        if node.move == "PASS" and node.parent is not None and node.parent.move == "PASS":
            return True
        return False
    
    def _terminal_value(self, state, to_play):
        black_score = state.score(1)
        white_score = state.score(2)
        if black_score > white_score + state.komi:
            winner = 1
        elif black_score < white_score + state.komi:
            winner = 2
        else:
            winner = 0
        
        if winner == 0:
            return 0.0
        return 1.0 if winner == to_play else -1.0
    
    def _apply_move(self, state, move, piece_type):
        next_state = state.copy()
        if move == "PASS":
            next_state.previous_board = deepcopy(next_state.board)
            next_state.n_move += 1
            next_state.X_move = not next_state.X_move
            return next_state
        
        next_state.place_chess(move[0], move[1], piece_type)
        return next_state
    
    def _expand(self, node, add_noise):
        legal_moves = get_valid_moves_mcts(node.state, node.to_play, self.move_cache)
        legal_moves = list(legal_moves)
        legal_moves.append("PASS")
        
        state_input = encode_board(node.state, node.to_play)
        policy, value = self.network.predict(state_input, self.device)
        
        priors = []
        for move in legal_moves:
            idx = self._action_to_index(move, node.state.size)
            priors.append(float(policy[idx]))
        priors = np.array(priors, dtype=np.float32)
        
        prior_sum = float(priors.sum())
        if prior_sum > 0:
            priors /= prior_sum
        else:
            priors.fill(1.0 / len(priors))
        
        if add_noise and len(priors) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors)).astype(np.float32)
            priors = (1.0 - self.dirichlet_eps) * priors + self.dirichlet_eps * noise
        
        for move, prior in zip(legal_moves, priors):
            child_state = self._apply_move(node.state, move, node.to_play)
            child = AlphaZeroNode(
                parent=node,
                move=move,
                prior=float(prior),
                state=child_state,
                to_play=3 - node.to_play,
            )
            node.children[move] = child
        
        node.is_expanded = True
        return float(value)
    
    def _backpropagate(self, search_path, value):
        v = float(value)
        for node in reversed(search_path):
            node.visits += 1
            node.value_sum += v
            v = -v
    
    def search(self, go, piece_type, temperature=1.0, add_noise=False):
        temperature = float(temperature)
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        self.move_cache.clear()
        root = AlphaZeroNode(state=MCTSState(go), to_play=piece_type, prior=1.0)
        
        if self._is_terminal(root):
            return {"PASS": 1.0}
        
        self._expand(root, add_noise=add_noise)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
                if node is None:
                    break
                search_path.append(node)
            
            if node is None:
                break
            
            if self._is_terminal(node):
                value = self._terminal_value(node.state, node.to_play)
                self._backpropagate(search_path, value)
                continue
            
            if not node.is_expanded:
                value = self._expand(node, add_noise=False)
                self._backpropagate(search_path, value)
                continue
            
            value = self._expand(node, add_noise=False)
            self._backpropagate(search_path, value)
        
        moves = list(root.children.keys())
        visits = np.array([root.children[m].visits for m in moves], dtype=np.float32)
        
        if visits.sum() <= 0:
            return {"PASS": 1.0}
        
        if temperature == 0:
            best_idx = int(np.argmax(visits))
            return {moves[best_idx]: 1.0}
        
        if temperature != 1.0:
            visits = np.power(visits, 1.0 / temperature)
        
        total_visits = visits.sum()
        if total_visits <= 0 or not np.isfinite(total_visits):
            best_idx = int(np.argmax(visits))
            return {moves[best_idx]: 1.0}
        probs = visits / total_visits
        if not np.all(np.isfinite(probs)):
            best_idx = int(np.argmax(visits))
            return {moves[best_idx]: 1.0}
        return {m: float(p) for m, p in zip(moves, probs)}
    
    def get_valid_moves(self, go, piece_type):
        """获取合法落子位置"""
        return get_valid_moves(go, piece_type)
    
    def get_best_move(self, go, piece_type, temperature=1.0):
        """
        获取最佳落子（考虑温度参数）
        """
        temperature = float(temperature)
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        pi = self.search(go, piece_type, temperature=temperature, add_noise=False)
        
        if not pi:
            return "PASS"
        
        moves = list(pi.keys())
        probs = list(pi.values())
        
        if temperature == 0:
            best_move = max(pi.items(), key=lambda x: x[1])[0]
            return best_move
        else:
            if temperature != 1.0:
                probs = [p ** (1.0 / temperature) for p in probs]
                total = sum(probs)
                probs = [p / total for p in probs]
            
            idx = np.random.choice(len(moves), p=probs)
            return moves[idx]
