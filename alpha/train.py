"""
AlphaZero训练脚本

包含:
- SelfPlayTrainer: 自我对弈训练器
- 训练循环和自我对弈逻辑
- 对手博弈数据生成
- 实时训练曲线绘制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
import time
import pickle
import glob
import matplotlib.pyplot as plt
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha.network import AlphaZeroNet, encode_board
from alpha.mcts import AlphaZeroMCTS
from alpha.players.random_player import RandomPlayer
from alpha.players.random_player_MCTS import MCTSPlayer as MCTSPlayerV1
from alpha.players.random_player_0309 import MCTSPlayer as MCTSPlayerV2
from alpha.players.xsh_player import my_player as XshPlayer
from host import GO
from copy import deepcopy


def _move_to_index(move, size):
    if move == "PASS":
        return size * size
    return move[0] * size + move[1]


def _augment_state_policy(state, policy, board_size=5):
    policy_grid = policy[: board_size * board_size].reshape(board_size, board_size)
    pass_prob = policy[board_size * board_size]
    augmented = []
    for k in range(4):
        s_rot = np.rot90(state, k, axes=(1, 2))
        p_rot = np.rot90(policy_grid, k)
        p = np.concatenate([p_rot.reshape(-1), np.array([pass_prob], dtype=np.float32)], axis=0)
        augmented.append((s_rot, p))
        
        s_flip = np.flip(s_rot, axis=2)
        p_flip = np.flip(p_rot, axis=1)
        p2 = np.concatenate([p_flip.reshape(-1), np.array([pass_prob], dtype=np.float32)], axis=0)
        augmented.append((s_flip, p2))
    return augmented


def _format_move(action):
    if action == "PASS":
        return "PASS"
    return f"({action[0]}, {action[1]})"


def _piece_name(piece_type):
    return "黑" if piece_type == 1 else "白"


def _winner_name(winner):
    if winner == 0:
        return "平局"
    return f"{_piece_name(winner)}胜"


def _board_to_text(go):
    symbols = {0: ".", 1: "X", 2: "O"}
    header = "      " + " ".join(str(i) for i in range(go.size))
    rows = []
    for i, row in enumerate(go.board):
        rows.append(f"    {i} " + " ".join(symbols.get(cell, "?") for cell in row))
    return "\n".join([header] + rows)


def _self_play_worker_run(args):
    state_dict = args["state_dict"]
    num_games = args["num_games"]
    num_simulations = args["num_simulations"]
    c_puct = args["c_puct"]
    seed = args["seed"]
    temperature = args["temperature"]
    temp_moves = args["temp_moves"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    network = AlphaZeroNet(board_size=5, num_res_blocks=4, num_channels=64)
    network.load_state_dict(state_dict)
    network.to(torch.device("cpu"))
    network.eval()

    game = SelfPlayGame(network, num_simulations=num_simulations, c_puct=c_puct, device=torch.device("cpu"))
    results = []
    for _ in range(num_games):
        training_data, winner = game.play_game(temperature=temperature, temp_moves=temp_moves, show_progress=False)
        results.append((training_data, winner))
    return results


class SelfPlayGame:
    """
    单局自我对弈游戏
    
    生成训练数据: (state, policy, value)
    """
    
    def __init__(self, network, num_simulations=800, c_puct=1.5, device=None):
        """
        初始化
        
        参数:
            network: 神经网络
            num_simulations: MCTS模拟次数
            c_puct: PUCT探索系数
            device: 计算设备
        """
        self.network = network
        self.device = device
        self.mcts = AlphaZeroMCTS(network, num_simulations, c_puct, device)
    
    def get_valid_moves(self, go, piece_type):
        """获取合法落子"""
        valid_moves = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    test_go = go.copy_board()
                    test_go.board[i][j] = piece_type
                    test_go.remove_died_pieces(3 - piece_type)
                    if not test_go.compare_board(test_go.board, go.previous_board):
                        valid_moves.append((i, j))
        return valid_moves
    
    def make_move(self, go, i, j, piece_type):
        """执行落子"""
        go.previous_board = deepcopy(go.board)
        go.board[i][j] = piece_type
        go.remove_died_pieces(3 - piece_type)
        go.n_move += 1
        go.X_move = not go.X_move
    
    def play_game(self, temperature=1.0, temp_moves=10, show_progress=True, show_board=False, game_tag=""):
        """
        进行一局自我对弈
        
        参数:
            temperature: 温度参数 (控制探索)
        
        返回:
            list: [(state, policy, current_player), ...] 训练数据
            int: 获胜者 (0=平局, 1=黑胜, 2=白胜)
        """
        go = GO(5)
        go.init_board(5)
        go.history_boards = [deepcopy(go.board)]
        
        game_history = []
        current_piece = 1
        move_count = 0
        max_moves = 45
        last_move_pass = False
        
        while move_count < max_moves:
            if show_progress:
                prefix = f"    [SelfPlay {game_tag}] " if game_tag else "    "
                print(f"{prefix}第{move_count + 1}/{max_moves}手 | {_piece_name(current_piece)}方搜索中...")
            temp = temperature if move_count < temp_moves else 0
            state = encode_board(go, current_piece)
            pi = self.mcts.search(go, current_piece, temperature=temp, add_noise=True)
            
            # 记录训练数据
            # 将pi转换为固定大小的数组
            policy_target = np.zeros(26, dtype=np.float32)
            for move, prob in pi.items():
                idx = _move_to_index(move, 5)
                policy_target[idx] = prob
            
            game_history.append((state, policy_target, current_piece))
            
            moves = list(pi.keys())
            probs = list(pi.values())
            probs = np.array(probs, dtype=np.float64)
            probs_sum = probs.sum()
            if probs_sum <= 0 or not np.isfinite(probs_sum):
                probs = np.ones(len(probs), dtype=np.float64) / len(probs)
            else:
                probs = probs / probs_sum
            action = moves[np.random.choice(len(moves), p=probs)]
            
            # 执行动作
            if action == "PASS":
                go.previous_board = deepcopy(go.board)
                go.n_move += 1
                go.X_move = not go.X_move
                if last_move_pass:
                    move_count += 1
                    if show_progress:
                        print(f"      {_piece_name(current_piece)}方连续PASS，终局触发")
                    break
                last_move_pass = True
            else:
                self.make_move(go, action[0], action[1], current_piece)
                last_move_pass = False
            if show_progress:
                print(f"      {_piece_name(current_piece)}方落子: {_format_move(action)}")
            
            go.history_boards.append(deepcopy(go.board))
            if show_progress and show_board:
                print(_board_to_text(go))
            
            move_count += 1
            current_piece = 3 - current_piece

        if show_progress:
            print(f"    对局结束: 共{move_count}手")

        # 计算游戏结果
        winner = go.judge_winner()
        if show_progress:
            print(f"    对局结果: {_winner_name(winner)}")
        
        # 为每一步分配价值
        training_data = []
        for state, policy, player in game_history:
            if winner == 0:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            for aug_state, aug_policy in _augment_state_policy(state, policy, board_size=5):
                training_data.append((aug_state, aug_policy, value))
        
        return training_data, winner


class OpponentGame:
    """
    与对手博弈生成训练数据
    
    支持与 RandomPlayer、MCTSPlayerV1、MCTSPlayerV2 对弈
    """
    
    def __init__(self, network, num_simulations=800, c_puct=1.5, device=None):
        self.network = network
        self.device = device
        self.mcts = AlphaZeroMCTS(network, num_simulations, c_puct, device)
    
    def get_valid_moves(self, go, piece_type):
        valid_moves = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    test_go = go.copy_board()
                    test_go.board[i][j] = piece_type
                    test_go.remove_died_pieces(3 - piece_type)
                    if not test_go.compare_board(test_go.board, go.previous_board):
                        valid_moves.append((i, j))
        return valid_moves
    
    def make_move(self, go, i, j, piece_type):
        go.previous_board = deepcopy(go.board)
        go.board[i][j] = piece_type
        go.remove_died_pieces(3 - piece_type)
        go.n_move += 1
        go.X_move = not go.X_move
    
    def play_against_opponent(self, opponent, our_piece_type=1, temperature=0.5, show_progress=True, show_board=False, game_tag=""):
        """
        与对手进行一局对弈
        
        参数:
            opponent: 对手玩家实例
            our_piece_type: 我方棋子类型 (1=黑子先手, 2=白子后手)
            temperature: 温度参数
        
        返回:
            list: 训练数据
            int: 获胜者 (0=平局, 1=黑胜, 2=白胜)
        """
        go = GO(5)
        go.init_board(5)
        
        game_history = []
        current_piece = 1
        move_count = 0
        max_moves = 45
        last_move_pass = False
        
        while move_count < max_moves:
            if show_progress:
                role_name = "我方" if current_piece == our_piece_type else "对手"
                print(f"      [对手局 {game_tag}] 第{move_count + 1}/{max_moves}手 | {role_name}({_piece_name(current_piece)})搜索中...")
            
            valid_moves = self.get_valid_moves(go, current_piece)
            
            if not valid_moves:
                opp_moves = self.get_valid_moves(go, 3 - current_piece)
                if not opp_moves:
                    break
                current_piece = 3 - current_piece
                move_count += 1
                continue
            
            if current_piece == our_piece_type:
                state = encode_board(go, current_piece)
                pi = self.mcts.search(go, current_piece, temperature=temperature, add_noise=False)
                
                policy_target = np.zeros(26, dtype=np.float32)
                for move, prob in pi.items():
                    idx = _move_to_index(move, 5)
                    policy_target[idx] = prob
                
                game_history.append((state, policy_target, current_piece))
                
                moves = list(pi.keys())
                probs = list(pi.values())
                probs = np.array(probs, dtype=np.float64)
                probs_sum = probs.sum()
                if probs_sum <= 0 or not np.isfinite(probs_sum):
                    probs = np.ones(len(probs), dtype=np.float64) / len(probs)
                else:
                    probs = probs / probs_sum
                action = moves[np.random.choice(len(moves), p=probs)]
                
                if action == "PASS":
                    go.previous_board = deepcopy(go.board)
                    go.n_move += 1
                    go.X_move = not go.X_move
                    if last_move_pass:
                        move_count += 1
                        if show_progress:
                            print(f"      {_piece_name(current_piece)}方连续PASS，终局触发")
                        break
                    last_move_pass = True
                else:
                    self.make_move(go, action[0], action[1], current_piece)
                    last_move_pass = False
            else:
                # 获取对手合法落子以备校验
                valid_moves_for_opp = self.get_valid_moves(go, current_piece)
                action = opponent.get_input(go, current_piece)
                
                if action == "PASS" or action not in valid_moves_for_opp:
                    go.previous_board = deepcopy(go.board)
                    go.n_move += 1
                    go.X_move = not go.X_move
                    action = "PASS"
                    if last_move_pass:
                        move_count += 1
                        if show_progress:
                            print(f"      {_piece_name(current_piece)}方连续PASS，终局触发")
                        break
                    last_move_pass = True
                else:
                    self.make_move(go, action[0], action[1], current_piece)
                    last_move_pass = False
            if show_progress:
                role_name = "我方" if current_piece == our_piece_type else "对手"
                print(f"        {role_name}({_piece_name(current_piece)})落子: {_format_move(action)}")
            if show_progress and show_board:
                print(_board_to_text(go))
            
            move_count += 1
            current_piece = 3 - current_piece
        
        winner = go.judge_winner()
        if show_progress:
            print(f"      对局结果: {_winner_name(winner)}")
        
        training_data = []
        for state, policy, player in game_history:
            if winner == 0:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            for aug_state, aug_policy in _augment_state_policy(state, policy, board_size=5):
                training_data.append((aug_state, aug_policy, value))
        
        return training_data, winner


class TrainingLogger:
    """
    训练日志记录和曲线绘制
    """
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.iterations = []
        self.losses = []
        self.policy_entropies = []
        self.win_rates_random = []
        self.win_rates_mcts_v1 = []
        self.win_rates_mcts_v2 = []
        self.win_rates_xsh = []
        self.buffer_sizes = []
        
        self.log_file = os.path.join(save_dir, "training_log.txt")
        self.curve_file = os.path.join(save_dir, "training_curves.png")
    
    def log(self, iteration, loss, policy_entropy, buffer_size, win_rates=None):
        self.iterations.append(iteration)
        self.losses.append(loss)
        self.policy_entropies.append(policy_entropy)
        self.buffer_sizes.append(buffer_size)
        
        if win_rates:
            self.win_rates_random.append(win_rates.get('random', 0))
            self.win_rates_mcts_v1.append(win_rates.get('mcts_v1', 0))
            self.win_rates_mcts_v2.append(win_rates.get('mcts_v2', 0))
            self.win_rates_xsh.append(win_rates.get('xsh', 0))
        
        with open(self.log_file, 'a') as f:
            f.write(f"Iteration {iteration}: Loss={loss:.4f}, PolicyEntropy={policy_entropy:.4f}, Buffer={buffer_size}")
            if win_rates:
                f.write(f", WinRates={win_rates}")
            f.write("\n")
    
    def plot_curves(self):
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        
        if self.iterations and self.losses:
            ax1 = axes[0, 0]
            ax1.plot(self.iterations, self.losses, 'b-', linewidth=2)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.grid(True, alpha=0.3)
        
        if self.iterations and self.buffer_sizes:
            ax2 = axes[0, 1]
            ax2.plot(self.iterations, self.buffer_sizes, 'g-', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Buffer Size')
            ax2.set_title('Training Buffer Size')
            ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        if self.win_rates_random:
            ax3.plot(self.iterations[-len(self.win_rates_random):], self.win_rates_random, 'r-', label='vs Random', linewidth=2)
        if self.win_rates_mcts_v1:
            ax3.plot(self.iterations[-len(self.win_rates_mcts_v1):], self.win_rates_mcts_v1, 'b-', label='vs MCTS V1', linewidth=2)
        if self.win_rates_mcts_v2:
            ax3.plot(self.iterations[-len(self.win_rates_mcts_v2):], self.win_rates_mcts_v2, 'g-', label='vs MCTS V2', linewidth=2)
        if self.win_rates_xsh:
            ax3.plot(self.iterations[-len(self.win_rates_xsh):], self.win_rates_xsh, 'm-', label='vs Xsh', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Win Rate')
        ax3.set_title('Win Rates vs Opponents')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        ax4 = axes[1, 1]
        if self.iterations and self.policy_entropies:
            ax4.plot(self.iterations, self.policy_entropies, color='orange', linewidth=2)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Policy Entropy')
            ax4.set_title('Policy Entropy')
            ax4.grid(True, alpha=0.3)
        
        ax5 = axes[2, 0]
        if self.losses and len(self.losses) > 1:
            recent_losses = self.losses[-min(20, len(self.losses)):]
            recent_iters = self.iterations[-min(20, len(self.iterations)):]
            ax5.plot(recent_iters, recent_losses, 'b-', linewidth=2, marker='o')
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('Loss')
            ax5.set_title('Recent Training Loss (Last 20)')
            ax5.grid(True, alpha=0.3)
        
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.curve_file, dpi=150)
        plt.close()
        
        return self.curve_file


class SelfPlayTrainer:
    """
    AlphaZero自我对弈训练器
    
    管理训练循环、模型保存和评估
    """
    
    def __init__(self, num_games_per_iteration=100, num_simulations=800,
                 learning_rate=0.001, batch_size=32, epochs=10,
                 buffer_size=100000,
                 use_existing_data=False, merge_data=True,
                 load_existing_model=True, strict_az_mode=False,
                 lr_decay_enabled=True, lr_decay_steps=30, lr_decay_rate=0.5,
                 play_self_play=True,
                 play_vs_random=False, random_games=0,
                 play_vs_mcts_v1=False, mcts_v1_games=0,
                 play_vs_mcts_v2=False, mcts_v2_games=0,
                 play_vs_xsh=False, xsh_games=0,
                 num_selfplay_workers=1, selfplay_temperature=1.0, selfplay_temp_moves=10,
                 verbose_progress=True, show_game_process=True, show_board=False):
        """
        初始化训练器

        参数:
            num_games_per_iteration: 每次迭代自我对弈局数
            num_simulations: MCTS模拟次数
            learning_rate: 学习率
            batch_size: 批量大小
            epochs: 每轮训练epoch数
            buffer_size: 训练数据缓冲区最大容量
            use_existing_data: 是否使用已有数据集
            merge_data: 是否合并新旧数据集 (仅当use_existing_data=True时有效)
            load_existing_model: 是否加载已有模型继续训练
            strict_az_mode: 严格AlphaZero模式，每轮训练后清空Buffer
            lr_decay_enabled: 是否启用学习率衰减
            lr_decay_steps: 学习率衰减间隔（每N轮衰减一次）
            lr_decay_rate: 学习率衰减率
            play_self_play: 是否进行自我对弈生成数据
            play_vs_random: 是否与RandomPlayer对弈生成数据
            random_games: 与RandomPlayer对弈的局数
            play_vs_mcts_v1: 是否与MCTSPlayerV1对弈生成数据
            mcts_v1_games: 与MCTSPlayerV1对弈的局数
            play_vs_mcts_v2: 是否与MCTSPlayerV2对弈生成数据
            mcts_v2_games: 与MCTSPlayerV2对弈的局数
            play_vs_xsh: 是否与XshPlayer对弈生成数据
            xsh_games: 与XshPlayer对弈的局数
            num_selfplay_workers: 自我对弈并行进程数
            selfplay_temperature: 自我对弈温度参数
            selfplay_temp_moves: 温度采样步数
            verbose_progress: 是否显示训练详细进度
            show_game_process: 是否显示对局过程中的落子信息
            show_board: 是否显示每步后的棋盘状态
        """
        self.num_games_per_iteration = num_games_per_iteration
        self.num_simulations = num_simulations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_existing_data = use_existing_data
        self.merge_data = merge_data
        self.load_existing_model = load_existing_model
        self.strict_az_mode = strict_az_mode
        self.lr_decay_enabled = lr_decay_enabled
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.initial_lr = learning_rate
        self.play_self_play = play_self_play
        self.play_vs_random = play_vs_random
        self.random_games = random_games
        self.play_vs_mcts_v1 = play_vs_mcts_v1
        self.mcts_v1_games = mcts_v1_games
        self.play_vs_mcts_v2 = play_vs_mcts_v2
        self.mcts_v2_games = mcts_v2_games
        self.play_vs_xsh = play_vs_xsh
        self.xsh_games = xsh_games
        self.num_selfplay_workers = max(1, int(num_selfplay_workers))
        self.selfplay_temperature = float(selfplay_temperature)
        self.selfplay_temp_moves = int(selfplay_temp_moves)
        self.verbose_progress = bool(verbose_progress)
        self.show_game_process = bool(show_game_process)
        self.show_board = bool(show_board)
        if self.selfplay_temperature < 0:
            raise ValueError(f"selfplay_temperature must be >= 0, got {self.selfplay_temperature}")
        if self.selfplay_temp_moves < 0:
            raise ValueError(f"selfplay_temp_moves must be >= 0, got {self.selfplay_temp_moves}")
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        self.network = AlphaZeroNet(board_size=5, num_res_blocks=4, num_channels=64)
        self.network.to(self.device)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

        self.buffer_size = buffer_size
        self.training_buffer = deque(maxlen=buffer_size)
        
        alpha_dir = os.path.dirname(__file__)
        self.model_dir = os.path.join(alpha_dir, "models")
        self.data_dir = os.path.join(alpha_dir, "training_data")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger = TrainingLogger(alpha_dir)
        
        self.iteration = 0
        self.total_games = 0
        self.data_file_counter = 0
        
        self.opponents = {}
        self.opponent_games = {}
        if play_vs_random and random_games > 0:
            self.opponents['random'] = RandomPlayer()
            self.opponent_games['random'] = random_games
        if play_vs_mcts_v1 and mcts_v1_games > 0:
            self.opponents['mcts_v1'] = MCTSPlayerV1(num_simulations=200, timeout=5)
            self.opponent_games['mcts_v1'] = mcts_v1_games
        if play_vs_mcts_v2 and mcts_v2_games > 0:
            self.opponents['mcts_v2'] = MCTSPlayerV2()
            self.opponent_games['mcts_v2'] = mcts_v2_games
        if play_vs_xsh and xsh_games > 0:
            self.opponents['xsh'] = XshPlayer()
            self.opponent_games['xsh'] = xsh_games

    def run_self_play_parallel(self, iteration):
        game_results = {0: 0, 1: 0, 2: 0}
        start_time = time.time()
        completed_games = 0
        preview_games = self.num_games_per_iteration if self.show_game_process else 0
        if self.show_game_process and preview_games > 0:
            print(f"  先展示 {preview_games} 局详细自博弈过程")
            for _ in range(preview_games):
                game = SelfPlayGame(self.network, self.num_simulations, device=self.device)
                training_data, winner = game.play_game(
                    temperature=self.selfplay_temperature,
                    temp_moves=self.selfplay_temp_moves,
                    show_progress=self.verbose_progress,
                    show_board=self.show_board,
                    game_tag=f"{completed_games + 1}/{self.num_games_per_iteration}"
                )
                self.save_training_data(training_data, iteration, completed_games)
                self.training_buffer.extend(training_data)
                game_results[winner] += 1
                self.total_games += 1
                completed_games += 1
                elapsed = time.time() - start_time
                percent = completed_games / self.num_games_per_iteration * 100
                print(
                    f"  [SelfPlay] {completed_games}/{self.num_games_per_iteration} ({percent:.1f}%) | "
                    f"B={game_results[1]} W={game_results[2]} D={game_results[0]} | "
                    f"Buffer={len(self.training_buffer)} | 用时={elapsed:.1f}s"
                )

        remaining_games = self.num_games_per_iteration - completed_games
        if remaining_games <= 0:
            return game_results

        if self.verbose_progress:
            worker_count = 1
        else:
            worker_count = min(self.num_selfplay_workers, remaining_games)
        if worker_count <= 1:
            for _ in range(remaining_games):
                game = SelfPlayGame(self.network, self.num_simulations, device=self.device)
                training_data, winner = game.play_game(
                    temperature=self.selfplay_temperature,
                    temp_moves=self.selfplay_temp_moves,
                    show_progress=self.verbose_progress,
                    show_board=self.show_board
                )
                self.save_training_data(training_data, iteration, completed_games)
                self.training_buffer.extend(training_data)
                game_results[winner] += 1
                self.total_games += 1
                completed_games += 1
                elapsed = time.time() - start_time
                percent = completed_games / self.num_games_per_iteration * 100
                print(
                    f"  [SelfPlay] {completed_games}/{self.num_games_per_iteration} ({percent:.1f}%) | "
                    f"B={game_results[1]} W={game_results[2]} D={game_results[0]} | "
                    f"Buffer={len(self.training_buffer)} | 用时={elapsed:.1f}s"
                )
            return game_results

        state_dict_cpu = {k: v.detach().cpu() for k, v in self.network.state_dict().items()}
        base_seed = int(time.time()) % 1000000
        counts = [remaining_games // worker_count] * worker_count
        for i in range(remaining_games % worker_count):
            counts[i] += 1

        futures = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for worker_id, game_count in enumerate(counts):
                if game_count <= 0:
                    continue
                args = {
                    "state_dict": state_dict_cpu,
                    "num_games": game_count,
                    "num_simulations": self.num_simulations,
                    "c_puct": 1.5,
                    "seed": base_seed + worker_id * 9973,
                    "temperature": self.selfplay_temperature,
                    "temp_moves": self.selfplay_temp_moves,
                }
                futures.append(executor.submit(_self_play_worker_run, args))

            for future in as_completed(futures):
                results = future.result()
                for training_data, winner in results:
                    self.save_training_data(training_data, iteration, completed_games)
                    self.training_buffer.extend(training_data)
                    game_results[winner] += 1
                    self.total_games += 1
                    completed_games += 1
                    elapsed = time.time() - start_time
                    percent = completed_games / self.num_games_per_iteration * 100
                    print(
                        f"  [SelfPlay] {completed_games}/{self.num_games_per_iteration} ({percent:.1f}%) | "
                        f"B={game_results[1]} W={game_results[2]} D={game_results[0]} | "
                        f"Buffer={len(self.training_buffer)} | 用时={elapsed:.1f}s"
                    )
        return game_results
    
    def train_network(self):
        """
        训练神经网络
        
        返回:
            tuple: (平均损失, 平均策略熵)
        """
        if len(self.training_buffer) < self.batch_size:
            print(f"Not enough training data: {len(self.training_buffer)} < {self.batch_size}")
            return 0.0, 0.0
        
        self.network.train()
        total_loss = 0.0
        total_policy_entropy = 0.0
        num_batches = 0
        training_start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            indices = np.random.choice(len(self.training_buffer), 
                                      min(self.batch_size * 10, len(self.training_buffer)), 
                                      replace=False)
            total_epoch_batches = max(1, math.ceil(len(indices) / self.batch_size))
            print(f"  Epoch {epoch + 1}/{self.epochs} | Batches={total_epoch_batches}")
            
            epoch_loss_sum = 0.0
            epoch_entropy_sum = 0.0
            epoch_batch_count = 0
            progress_interval = max(1, total_epoch_batches // 10)
            for batch_idx, i in enumerate(range(0, len(indices), self.batch_size), start=1):
                batch_indices = indices[i:i + self.batch_size]
                batch = [self.training_buffer[idx] for idx in batch_indices]
                
                # 准备数据并移动到GPU
                states = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
                target_pis = torch.FloatTensor(np.array([x[1] for x in batch])).to(self.device)
                target_vs = torch.FloatTensor(np.array([x[2] for x in batch])).to(self.device)
                
                # 前向传播
                pred_pis, pred_vs = self.network(states)
                
                # 计算损失
                # 策略损失 (交叉熵)
                policy_loss = -torch.sum(target_pis * pred_pis) / len(batch_indices)
                # 价值损失 (MSE)
                value_loss = F.mse_loss(pred_vs.squeeze(), target_vs)
                # 总损失
                loss = policy_loss + value_loss
                policy_entropy = -(pred_pis.exp() * pred_pis).sum(dim=1).mean()
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_entropy += policy_entropy.item()
                num_batches += 1
                epoch_loss_sum += loss.item()
                epoch_entropy_sum += policy_entropy.item()
                epoch_batch_count += 1
                if batch_idx % progress_interval == 0 or batch_idx == total_epoch_batches:
                    avg_loss_so_far = epoch_loss_sum / epoch_batch_count
                    avg_entropy_so_far = epoch_entropy_sum / epoch_batch_count
                    print(
                        f"    Batch {batch_idx}/{total_epoch_batches} | "
                        f"Loss={avg_loss_so_far:.4f} | PolicyEntropy={avg_entropy_so_far:.4f}"
                    )
            epoch_elapsed = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss_sum / epoch_batch_count if epoch_batch_count > 0 else 0.0
            avg_epoch_entropy = epoch_entropy_sum / epoch_batch_count if epoch_batch_count > 0 else 0.0
            print(
                f"  Epoch {epoch + 1} 完成 | AvgLoss={avg_epoch_loss:.4f} | "
                f"AvgEntropy={avg_epoch_entropy:.4f} | 用时={epoch_elapsed:.1f}s"
            )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_policy_entropy = total_policy_entropy / num_batches if num_batches > 0 else 0.0
        total_elapsed = time.time() - training_start_time
        print(f"  训练阶段完成 | 总Batch={num_batches} | 总用时={total_elapsed:.1f}s")
        return avg_loss, avg_policy_entropy
    
    def save_model(self, filename=None):
        """
        保存模型
        
        参数:
            filename: 文件名 (默认使用迭代次数)
        """
        if filename is None:
            filename = f"model_iter_{self.iteration:03d}.pth"
        
        filepath = os.path.join(self.model_dir, filename)
        torch.save({
            'iteration': self.iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_games': self.total_games,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def save_best_model(self):
        best_path = os.path.join(os.path.dirname(__file__), "model_best.pth")
        torch.save(self.network.state_dict(), best_path)
    
    def load_model(self, filepath):
        """
        加载模型
        
        参数:
            filepath: 模型文件路径
        """
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, weights_only=False)
            if 'model_state_dict' in checkpoint:
                try:
                    self.network.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.iteration = checkpoint.get('iteration', -1) + 1
                    self.total_games = checkpoint.get('total_games', 0)
                except Exception as e:
                    print(f"Warning: Failed to load checkpoint weights: {e}")
                    return False
            else:
                try:
                    self.network.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"Warning: Failed to load state_dict: {e}")
                    return False
            print(f"Model loaded from {filepath}")
            return True
        return False
    
    def save_training_data(self, data, iteration, game_idx):
        """
        保存训练数据到文件
        
        参数:
            data: 训练数据列表 [(state, policy, value), ...]
            iteration: 当前迭代次数
            game_idx: 当前游戏索引 (整数或字符串)
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"data_{timestamp}.pkl"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"    Training data saved to {filepath}")
        return filepath
    
    def load_training_data(self, filepath):
        """
        从文件加载训练数据
        
        参数:
            filepath: 数据文件路径
        
        返回:
            list: 训练数据列表
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        filtered = []
        for item in data:
            try:
                s, p, v = item
                if np.shape(s) == (4, 5, 5) and np.shape(p) == (26,):
                    filtered.append(item)
            except:
                continue
        print(f"    Loaded {len(filtered)} samples from {filepath}")
        return filtered
    
    def load_all_training_data(self):
        """
        加载所有训练数据文件，按时间顺序从旧到新加载，
        buffer满时自动淘汰旧数据

        返回:
            int: 加载的样本总数
        """
        pattern = os.path.join(self.data_dir, "data_*.pkl")
        files = sorted(glob.glob(pattern))

        if not files:
            print("    No existing training data found.")
            return 0

        total_samples = 0
        loaded_samples = 0
        for filepath in files:
            try:
                data = self.load_training_data(filepath)
                self.training_buffer.extend(data)
                loaded_samples = len(data)
                total_samples += loaded_samples
            except Exception as e:
                print(f"    Warning: Failed to load {filepath}: {e}")

        print(f"    Total loaded: {total_samples} samples from {len(files)} files")
        print(f"    Buffer size: {len(self.training_buffer)} (max: {self.buffer_size})")
        return total_samples

    def get_existing_data_info(self):
        """
        获取已有数据集信息

        返回:
            dict: 包含文件数量、样本总数等信息
        """
        pattern = os.path.join(self.data_dir, "data_*.pkl")
        files = sorted(glob.glob(pattern))
        
        total_samples = 0
        for filepath in files:
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    total_samples += len(data)
            except:
                pass
        
        return {
            'file_count': len(files),
            'total_samples': total_samples,
            'files': files
        }
    
    def play_against_opponents(self, iteration):
        """
        与对手博弈生成训练数据
        
        参数:
            iteration: 当前迭代次数
        
        返回:
            dict: 各对手的胜率统计
        """
        if not self.opponents:
            return {}
        
        win_rates = {}
        opponent_game = OpponentGame(self.network, self.num_simulations, device=self.device)
        
        for opp_name, opponent in self.opponents.items():
            print(f"\n  Playing against {opp_name}...")
            wins = 0
            losses = 0
            draws = 0
            total = self.opponent_games.get(opp_name, 0)
            opp_start_time = time.time()
            
            for game_idx in range(total):
                our_piece_type = 1 if game_idx % 2 == 0 else 2
                detailed = self.show_game_process and game_idx == 0
                training_data, winner = opponent_game.play_against_opponent(
                    opponent,
                    our_piece_type,
                    temperature=0.5,
                    show_progress=detailed,
                    show_board=(self.show_board and detailed),
                    game_tag=f"{game_idx + 1}/{total}"
                )
                
                self.save_training_data(training_data, iteration, 
                                       f"{opp_name}_{game_idx}")
                self.training_buffer.extend(training_data)
                self.total_games += 1
                
                if winner == 0:
                    draws += 1
                elif (winner == 1 and our_piece_type == 1) or (winner == 2 and our_piece_type == 2):
                    wins += 1
                else:
                    losses += 1
                current_count = game_idx + 1
                current_win_rate = wins / current_count if current_count > 0 else 0.0
                elapsed = time.time() - opp_start_time
                progress = current_count / total * 100 if total > 0 else 100.0
                print(
                    f"    {opp_name}: {current_count}/{total} ({progress:.1f}%) | "
                    f"Win={wins} Lose={losses} Draw={draws} | "
                    f"WinRate={current_win_rate:.1%} | "
                    f"Buffer={len(self.training_buffer)} | 用时={elapsed:.1f}s"
                )
            
            win_rate = wins / total if total > 0 else 0
            win_rates[opp_name] = win_rate
            print(f"  {opp_name} Win Rate: {win_rate:.2%} ({wins}W/{losses}L/{draws}D)")
        
        return win_rates

    def _play_eval_game(self, network_black, network_white, num_simulations):
        go = GO(5)
        go.init_board(5)
        go.history_boards = [deepcopy(go.board)]
        mcts_black = AlphaZeroMCTS(network_black, num_simulations, device=self.device)
        mcts_white = AlphaZeroMCTS(network_white, num_simulations, device=self.device)
        current_piece = 1
        last_move_pass = False
        max_moves = 25

        for _ in range(max_moves):
            if current_piece == 1:
                action = mcts_black.get_best_move(go, 1, temperature=0)
            else:
                action = mcts_white.get_best_move(go, 2, temperature=0)
            
            if action == "PASS":
                go.previous_board = deepcopy(go.board)
                go.n_move += 1
                go.X_move = not go.X_move
                if last_move_pass:
                    break
                last_move_pass = True
            else:
                go.previous_board = deepcopy(go.board)
                go.board[action[0]][action[1]] = current_piece
                go.remove_died_pieces(3 - current_piece)
                go.n_move += 1
                go.X_move = not go.X_move
                last_move_pass = False
            
            go.history_boards.append(deepcopy(go.board))
            current_piece = 3 - current_piece

        return go.judge_winner()

    def evaluate_and_update_best(self, num_games=10, num_simulations=200, win_rate_threshold=0.55):
        best_path = os.path.join(os.path.dirname(__file__), "model_best.pth")
        if not os.path.exists(best_path):
            self.save_best_model()
            return 1.0
        
        best_network = AlphaZeroNet(board_size=5, num_res_blocks=4, num_channels=64).to(self.device)
        best_network.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
        best_network.eval()
        self.network.eval()

        wins = 0
        losses = 0
        draws = 0
        for game_idx in range(num_games):
            candidate_black = (game_idx % 2 == 0)
            if candidate_black:
                winner = self._play_eval_game(self.network, best_network, num_simulations)
                if winner == 1:
                    wins += 1
                elif winner == 2:
                    losses += 1
                else:
                    draws += 1
            else:
                winner = self._play_eval_game(best_network, self.network, num_simulations)
                if winner == 2:
                    wins += 1
                elif winner == 1:
                    losses += 1
                else:
                    draws += 1

        win_rate = wins / num_games if num_games > 0 else 0.0
        if win_rate >= win_rate_threshold:
            self.save_best_model()
        return win_rate
    
    def run_training_loop(self, num_iterations=100):
        """
        运行完整训练循环
        
        参数:
            num_iterations: 训练迭代次数
        """
        print("=" * 60)
        print("AlphaZero Training Started")
        print("=" * 60)
        print(f"Iterations: {num_iterations}")
        print(f"MCTS simulations: {self.num_simulations}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"Use existing data: {self.use_existing_data}")
        print(f"Merge data: {self.merge_data}")
        print(f"Self-play: {self.play_self_play} ({self.num_games_per_iteration} games)")
        print(f"Self-play workers: {self.num_selfplay_workers}")
        print(f"Self-play temperature: {self.selfplay_temperature}, temp_moves: {self.selfplay_temp_moves}")
        print(f"Verbose progress: {self.verbose_progress}")
        print(f"Show game process: {self.show_game_process}")
        print(f"Show board: {self.show_board}")
        print(f"Play vs Random: {self.play_vs_random} ({self.random_games} games)")
        print(f"Play vs MCTS V1: {self.play_vs_mcts_v1} ({self.mcts_v1_games} games)")
        print(f"Play vs MCTS V2: {self.play_vs_mcts_v2} ({self.mcts_v2_games} games)")
        print("=" * 60)
        
        best_model_path = os.path.join(os.path.dirname(__file__), "model_best.pth")
        if self.load_existing_model and os.path.exists(best_model_path):
            self.load_model(best_model_path)
        elif not self.load_existing_model:
            print("Skipping existing model, training from scratch")

        allow_new_data_generation = (not self.use_existing_data) or self.merge_data
        
        if self.use_existing_data:
            data_info = self.get_existing_data_info()
            print(f"\n[Data] Found {data_info['file_count']} existing data files")
            print(f"[Data] Total existing samples: {data_info['total_samples']}")
            
            if data_info['file_count'] > 0:
                print("\n[Data] Loading existing training data...")
                self.load_all_training_data()
                
                if not self.merge_data:
                    print("[Data] Using ONLY existing data (no new data generation)")
        
        for iteration in range(self.iteration, num_iterations):
            self.iteration = iteration
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")
            
            if self.play_self_play and allow_new_data_generation:
                print("\n[1/5] Self-playing...")
                game_results = self.run_self_play_parallel(iteration)
                
                print(f"\nGame Results: Black={game_results[1]}, White={game_results[2]}, Draw={game_results[0]}")
            else:
                if not self.play_self_play:
                    print("\n[1/5] Skipping self-play (disabled)")
                else:
                    print("\n[1/5] Skipping self-play (merge_data=False uses only existing data)")
            
            if allow_new_data_generation:
                print("\n[2/5] Playing against opponents...")
                win_rates = self.play_against_opponents(iteration)
            else:
                print("\n[2/5] Skipping opponent games (merge_data=False uses only existing data)")
                win_rates = {}
            
            # 学习率衰减
            if self.lr_decay_enabled and iteration > 0 and iteration % self.lr_decay_steps == 0:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = param_group['lr'] * self.lr_decay_rate
                    print(f"  Learning rate decay: {old_lr:.6f} -> {param_group['lr']:.6f}")

            # 3. 训练网络
            print("\n[3/5] Training network...")
            avg_loss, avg_policy_entropy = self.train_network()
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Average Policy Entropy: {avg_policy_entropy:.4f}")
            
            # 4. 保存模型
            print("\n[4/5] Saving model...")
            self.save_model()
            best_win_rate = self.evaluate_and_update_best(num_games=10, num_simulations=200, win_rate_threshold=0.55)
            print(f"Best model update win rate: {best_win_rate:.2%}")
            
            # 5. 记录日志和绘制曲线
            print("\n[5/5] Logging and plotting...")
            self.logger.log(iteration + 1, avg_loss, avg_policy_entropy, len(self.training_buffer), win_rates)
            curve_path = self.logger.plot_curves()
            print(f"  Training curves saved to: {curve_path}")

            print(f"\n  Total games played: {self.total_games}")
            print(f"  Training buffer size: {len(self.training_buffer)}")

            if self.strict_az_mode:
                print("  [Strict AlphaZero Mode] Clearing training buffer...")
                self.training_buffer.clear()
                self.total_games = 0
        
        print("\n" + "=" * 60)
        print("Training Completed!")
        print(f"Final model saved to: {best_model_path}")
        print(f"Training curves saved to: {self.logger.save_dir}")
        print("=" * 60)


def main():
    """
    主函数
    
    数据集配置说明:
        use_existing_data=False, merge_data=True  (默认)
            - 不使用已有数据，只生成新数据进行训练
            
        use_existing_data=True, merge_data=True
            - 加载已有数据 + 生成新数据，合并后训练
            - 适合在已有数据基础上继续增强模型
            
        use_existing_data=True, merge_data=False
            - 只使用已有数据，不生成新数据
            - 适合使用固定数据集进行训练

    严格AlphaZero模式 (strict_az_mode):
        strict_az_mode=True
            - 每轮训练后清空Buffer，数据完全不累积
            - 每次迭代独立，完全遵循AlphaZero原始论文
            - Buffer最多保留 num_games_per_iteration 局产生的数据

    博弈配置说明:
        play_self_play: 是否进行自我对弈生成数据
        play_vs_random: 是否与随机玩家对弈
        random_games: 与随机玩家对弈局数
        play_vs_mcts_v1: 是否与MCTS V1玩家对弈
        mcts_v1_games: 与MCTS V1玩家对弈局数
        play_vs_mcts_v2: 是否与MCTS V2玩家对弈
        mcts_v2_games: 与MCTS V2玩家对弈局数
        play_vs_xsh: 是否与XshPlayer对弈
        xsh_games: 与XshPlayer对弈局数
    """
    torch.manual_seed(30)
    np.random.seed(30)
    # 设置随机种子确保训练结果可复现
    # 相同的种子会产生相同的随机序列，便于调试和对比实验
    
    trainer = SelfPlayTrainer(
        num_games_per_iteration=30,     # 每轮迭代自我对弈局数
        num_simulations=4000,            # MCTS模拟次数 (越大越准但越慢)
        learning_rate=0.001,            # 学习率
        batch_size=64,                  # 批量大小
        epochs=30,                      # 每轮训练epoch数
        buffer_size=60000,             # 训练数据缓冲区最大容量
        use_existing_data=True,         # 是否加载已有训练数据
        merge_data=True,                # 是否合并新旧数据 (False则只用新数据)
        load_existing_model=True,        # 是否加载已有模型继续训练
        strict_az_mode=False,            # 严格AlphaZero模式，每轮清空Buffer数据缓冲区
        lr_decay_enabled=False,         # 是否启用学习率衰减
        lr_decay_steps=15,              # 学习率衰减间隔（每N轮衰减一次）
        lr_decay_rate=0.5,             # 学习率衰减率
        play_self_play=True,            # 是否进行自我对弈生成数据
        play_vs_random=False,            # 是否与随机玩家对弈
        random_games=5,                 # 每轮与随机玩家对弈局数
        play_vs_mcts_v1=False,           # 是否与MCTS V1玩家对弈
        mcts_v1_games=5,                # 每轮与MCTS V1玩家对弈局数
        play_vs_mcts_v2=False,           # 是否与MCTS V2玩家对弈
        mcts_v2_games=20,               # 每轮与MCTS V2玩家对弈局数
        play_vs_xsh=False,               # 是否与XshPlayer对弈
        xsh_games=5,                    # 每轮与XshPlayer对弈局数
        num_selfplay_workers=12,        # 16逻辑核建议先用12个并行自博弈进程
        selfplay_temperature=1.0,       # 前temp_moves步采样温度
        selfplay_temp_moves=10,         # 前10步探索，后续temperature=0
        verbose_progress=True,             # 是否显示训练详细进度（设为True时强制串行执行）
        show_game_process=True,            # 是否显示对局过程中的落子信息
        show_board=True                   # 是否显示每步后的棋盘状态
    )
    
    trainer.run_training_loop(num_iterations=30)  # 总迭代轮数


if __name__ == "__main__":
    main()
