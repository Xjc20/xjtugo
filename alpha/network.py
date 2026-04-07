"""
AlphaZero神经网络定义

包含:
- ResidualBlock: 残差块
- AlphaZeroNet: 主网络（策略头+价值头）
- encode_board: 棋盘状态编码函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """
    残差块 - 用于提取深层特征
    
    结构:
        Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> Add -> ReLU
    """
    
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
        out += residual  # 残差连接
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """
    AlphaZero风格神经网络
    
    输入: (batch, 4, 5, 5) 的棋盘状态
    输出: 
        - policy: (batch, 26) 落子概率分布 (含PASS)
        - value: (batch, 1) 局面评估 [-1, 1]
    
    架构:
        输入卷积 -> 残差塔 -> 策略头 + 价值头
    """
    
    def __init__(self, board_size=5, num_res_blocks=4, num_channels=64):
        """
        初始化网络
        
        参数:
            board_size: 棋盘大小 (默认5)
            num_res_blocks: 残差块数量 (默认4)
            num_channels: 卷积通道数 (默认64)
        """
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size + 1
        
        # 输入卷积层 (4通道 -> 64通道)
        self.conv_input = nn.Conv2d(4, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # 残差塔
        self.res_tower = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # 策略头
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)
        
        # 价值头
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: (batch, 4, 5, 5) 棋盘状态
        
        返回:
            policy: (batch, 26) 对数概率
            value: (batch, 1) 局面评估
        """
        # 输入卷积
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # 残差塔
        for block in self.res_tower:
            x = block(x)
        
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)  # 对数概率
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # [-1, 1]
        
        return policy, value
    
    def predict(self, board_state, device=None):
        """
        预测单个棋盘状态
        
        参数:
            board_state: numpy array (4, 5, 5)
            device: 计算设备 (None则使用当前设备)
        
        返回:
            policy: numpy array (26,) 概率分布
            value: float 局面评估
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        with torch.no_grad():
            state_tensor = torch.FloatTensor(board_state).unsqueeze(0).to(device)
            policy_log, value = self.forward(state_tensor)
            policy = torch.exp(policy_log)
            return policy.cpu().numpy()[0], value.cpu().numpy()[0][0]


def encode_board(go, piece_type):
    """
    将棋盘状态编码为神经网络输入 (4, 5, 5)
    
    编码方案:
    - 通道0: 当前玩家棋子位置
    - 通道1: 对手玩家棋子位置
    - 通道2: 对手最近一步落子位置（仅一个1，其余为0）
    - 通道3: 当前玩家是否先手（黑棋=全1，白棋=全0）
    
    参数:
        go: GO类实例
        piece_type: 当前玩家棋子类型 (1=黑子, 2=白子)
    返回:
        numpy array (4, 5, 5)
    """
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
