"""
对手玩家模块

将新的player文件放入此文件夹即可在训练中使用

使用方法:
    from alpha.players.random_player import RandomPlayer
    from alpha.players.random_player_MCTS import MCTSPlayer as MCTSPlayerV1
    from alpha.players.random_player_0309 import MCTSPlayer as MCTSPlayerV2

添加新玩家:
    1. 将新的player文件复制到此文件夹
    2. 在train.py中添加导入语句
    3. 在SelfPlayTrainer中添加对应参数
"""
