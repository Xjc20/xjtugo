"""
AlphaZero风格围棋AI玩家

文件功能:
    基于AlphaZero算法的围棋AI玩家，使用神经网络+MCTS搜索
    
主要功能:
    1. MCTSPlayer类: 使用AlphaZero MCTS进行决策
    2. 神经网络评估: 策略网络+价值网络
    3. PUCT搜索: 结合神经网络先验概率的MCTS
    
算法流程:
    1. 神经网络评估当前局面，输出策略和价值
    2. MCTS使用PUCT公式进行搜索
    3. 选择访问次数最多的动作执行
    
参数说明:
    - piece_type: 棋子类型 (1=黑子X, 2=白子O)
    - num_simulations: MCTS模拟次数 (默认800)
    - c_puct: PUCT探索系数 (默认1.5)
    
使用方法:
    python my_player.py
    (从input.txt读取输入，输出到output.txt)
    
训练:
    python alpha/train.py
"""

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
from read import readInput
from write import writeOutput
from host import GO


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=5, num_res_blocks=4, num_channels=64):
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size + 1

        self.conv_input = nn.Conv2d(4, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        self.res_tower = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.res_tower:
            x = block(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def predict(self, board_state, device=None):
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        with torch.no_grad():
            state_tensor = torch.FloatTensor(board_state).unsqueeze(0).to(device)
            policy_log, value = self.forward(state_tensor)
            policy = torch.exp(policy_log)
            return policy.cpu().numpy()[0], value.cpu().numpy()[0][0]


def encode_board(go, piece_type):
    state = np.zeros((4, 5, 5), dtype=np.float32)
    board = np.array(go.board, dtype=np.float32)
    opponent = 3 - piece_type

    state[0] = (board == piece_type).astype(np.float32)
    state[1] = (board == opponent).astype(np.float32)

    previous_board = getattr(go, "previous_board", None)
    if previous_board is not None:
        previous_board = np.array(previous_board, dtype=np.float32)
        added = np.argwhere((previous_board != opponent) & (board == opponent))
        if len(added) > 0:
            x, y = added[0]
            state[2, x, y] = 1.0

    if piece_type == 1:
        state[3] = np.ones((5, 5), dtype=np.float32)

    return state


class MCTSState:
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
        for x, y in group:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx][ny] == 0:
                        return True
        return False

    def score(self, piece_type):
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == piece_type:
                    cnt += 1
        return cnt

    def _get_empty_group(self, i, j):
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
        return get_valid_moves(go, piece_type)

    def get_best_move(self, go, piece_type, temperature=1.0):
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


class MCTSPlayer:
    """
    AlphaZero风格MCTS围棋玩家
    
    使用神经网络指导的MCTS进行决策
    
    属性:
        network: AlphaZero神经网络
        mcts: AlphaZero MCTS搜索器
        num_simulations: MCTS模拟次数
        c_puct: PUCT探索系数
        expand_k: 每次扩展的子节点数 (来自0309)
        threshold: 提前终止阈值 (来自0309)
    """
    
    def __init__(self, model_path=None, num_simulations=800, c_puct=1.5, use_nn=True):
        """
        初始化MCTS玩家
        
        参数:
            model_path: 模型文件路径 (默认alpha/model_best.pth)
            num_simulations: MCTS模拟次数
            c_puct: PUCT探索系数
            use_nn: 是否使用神经网络 (False则使用纯MCTS)
            expand_k: 每次扩展的子节点数 (来自0309优化)
            threshold: 提前终止阈值 (来自0309优化)
        """
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.use_nn = use_nn
        
        # 检测CUDA是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}", file=sys.stderr)

        # 初始化神经网络
        self.network = AlphaZeroNet(board_size=5, num_res_blocks=4, num_channels=64)
        self.network.to(self.device)

        # 加载模型
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "alpha", "model_best.pth")

        if os.path.exists(model_path) and use_nn:
            try:
                self.network.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                self.network.eval()
                print(f"Model loaded from {model_path}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to load model: {e}", file=sys.stderr)
                print("Using random initialization", file=sys.stderr)
        else:
            if use_nn:
                print(f"Warning: Model not found at {model_path}", file=sys.stderr)
                print("Using random initialization", file=sys.stderr)

        # 创建MCTS搜索器
        self.mcts = AlphaZeroMCTS(
            self.network, 
            num_simulations, 
            c_puct, 
            self.device
        )
    
    def get_input(self, go, piece_type):
        """
        获取下一步落子位置
        
        参数:
            go: GO类实例，当前游戏状态
            piece_type: 玩家棋子类型 (1=黑子, 2=白子)
        
        返回:
            tuple: 落子位置 (row, col)
            str: "PASS" 如果没有合法落子位置
        """
        return self.mcts.get_best_move(go, piece_type, temperature=0)


if __name__ == "__main__":
    """
    主程序入口
    
    程序流程:
    1. 设置棋盘大小N=5（5x5围棋）
    2. 从input.txt读取当前游戏状态
    3. 创建GO实例并设置棋盘状态
    4. 创建MCTSPlayer实例（加载神经网络）
    5. 调用get_input()获取落子决策
    6. 将决策结果写入output.txt
    """
    
    # 初始化参数
    N = 5
    
    # 读取输入
    piece_type, previous_board, board = readInput(N)
    
    # 创建游戏实例
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    
    # 创建AlphaZero MCTS玩家
    # 参数: 模拟次数, c_puct
    player = MCTSPlayer(
        model_path="model_best.pth",
        num_simulations=600,
        c_puct=1.5,
        use_nn=True
    )
    
    # 获取下一步落子位置
    action = player.get_input(go, piece_type)
    
    # 输出结果
    writeOutput(action)
