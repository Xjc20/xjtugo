"""
AlphaZero围棋AI模块

包含:
- network: 神经网络定义
- mcts: PUCT搜索算法
- train: 训练脚本

使用方法:
    from alpha.network import AlphaZeroNet, encode_board
    from alpha.mcts import AlphaZeroMCTS
"""

from .network import AlphaZeroNet, encode_board
from .mcts import AlphaZeroMCTS, AlphaZeroNode

__all__ = ['AlphaZeroNet', 'encode_board', 'AlphaZeroMCTS', 'AlphaZeroNode']
__version__ = '1.0.0'
