# 西安交通大学电气工程学院人工智能导论围棋项目

## 项目概述

这是西安交通大学电气工程学院人工智能导论课程的围棋项目，实现了基于AlphaZero算法的围棋AI。项目包含完整的围棋游戏逻辑、GUI界面和AlphaZero训练框架。

## 游戏规则

本项目实现了标准的围棋规则，具体如下：

### 基本规则
- **棋盘大小**：5×5
- **棋子类型**：黑子(X)先行，白子(O)后行
- **落子规则**：
  - 只能落在空交叉点上
  - 不能自杀（落子后无气）
  - 不能重复前一回合的棋盘状态（KO规则）

### 胜负判定
- **提子规则**：当一方棋子被包围无气时，该棋子被提掉
- **计分方式**：计算双方剩余棋子数量
- **贴目规则**：白子获得 `n/2` 的贴目补偿（n为棋盘大小）
- **结束条件**：双方连续PASS或达到最大步数（n×n-1）

## 项目文件结构

```
xjtugo/
├── host.py              # 围棋游戏主机程序（命令行版）
├── host_gui.py          # 围棋游戏GUI界面
├── my_player.py         # AlphaZero AI玩家
├── human_gui_player.py  # 人类GUI玩家
├── test_gui.py          # GUI测试程序
├── read.py              # 输入读取模块
├── write.py             # 输出写入模块
├── clean.py             # 清理工具
├── check_cuda.py        # CUDA检测工具
├── build.sh             # Linux系统构建脚本
├── play_vs_with_host.bat # Windows系统AI对战脚本
├── play_vs_with_human.bat # Windows系统人机对战脚本
├── requirements.txt     # 项目依赖
├── model_best.pth       # 根目录最佳模型
├── AlphaZero_讲解.pptx  # AlphaZero算法讲解PPT
├── 智能围棋.pdf          # 智能围棋项目文档
├── init/                # 初始输入文件目录
│   └── input.txt
├── opponent/            # 对手玩家目录
│   ├── host.py
│   ├── read.py
│   ├── write.py
│   ├── random_player.py
│   ├── random_player_0309.py
│   ├── random_player_MCTS.py
│   ├── xsh_player.py
│   ├── my_player.xjy.py
│   ├── my_player.xjy3.py
│   └── my_player_xjy_final.py
└── alpha/               # AlphaZero算法实现
    ├── network.py       # 神经网络定义
    ├── mcts.py          # MCTS搜索算法
    ├── train.py         # 训练脚本
    ├── plot_training_curves.py # 训练曲线绘制脚本
    ├── model_best.pth   # alpha目录最佳模型
    ├── models/          # 模型保存目录（15个迭代版本）
    ├── training_log.txt # 训练日志文件
    ├── training_curves_new.png # 训练曲线图
    └── players/         # 其他玩家实现
        ├── random_player.py
        ├── random_player_0309.py
        ├── random_player_MCTS.py
        └── xsh_player.py
```

## 如何使用该项目

### 环境要求

```bash
pip install -r requirements.txt
```

主要依赖：
- torch>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0

### 运行方式

#### 1. Linux系统 - 自动测试模式
```bash
bash build.sh
```
- 自动与随机玩家进行4轮对战（黑子白子各2轮）
- 结果保存到 `cresult.txt` 文件

#### 2. Windows系统 - AI对战模式
```bash
# 默认：AlphaZero AI vs 随机玩家（命令行模式）
play_vs_with_host.bat

# 自定义参数：play_vs_with_host.bat [轮数] [我的玩家] [对手玩家] [主机程序]
# 示例：AlphaZero AI vs 优化版MCTS玩家（命令行模式）
play_vs_with_host.bat 4 my_player alpha/players/random_player_0309 host
```

#### 3. Windows系统 - 人机对战模式
```bash
# 默认：人类玩家 vs AlphaZero AI（GUI模式）
play_vs_with_human.bat
```


#### 4. 绘制训练曲线
```bash
python alpha/plot_training_curves.py
```
- 读取训练日志并生成训练曲线
- 结果保存为 `alpha/training_curves_new.png`


## AlphaZero算法原理

AlphaZero是一种结合深度学习和强化学习的算法，主要包含以下核心组件：

### 1. 自我对弈（Self-Play）
- AI与自己进行大量对局
- 记录每个局面的状态、策略和最终结果
- 使用温度参数控制探索程度（前10步使用较高温度，之后使用0温度）

### 2. 神经网络评估
- **策略网络**：预测每个可能动作的概率
- **价值网络**：评估当前局面的胜负概率
- 输入：4通道棋盘状态
- 输出：策略分布（26维）和价值估计（标量）

### 3. MCTS搜索（蒙特卡洛树搜索）
- 使用PUCT公式选择最优动作：
  ```
  U(s,a) = c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))
  Score = Q(s,a) + U(s,a)
  ```
- c_puct：探索系数（默认1.5）
- P(a|s)：神经网络先验概率
- N(s)：节点访问次数
- N(s,a)：动作访问次数
- Q(s,a)：动作平均价值

### 4. 训练流程
1. **数据生成**：自我对弈生成训练数据
2. **数据增强**：棋盘旋转和翻转（8种变换）
3. **模型训练**：使用策略损失和价值损失更新网络
4. **模型评估**：与不同对手对战，测试胜率
5. **模型保存**：保存每轮迭代的模型

## 网络结构

### 输入层
- 4通道 × 5 × 5 的棋盘状态：
  - 通道0：当前玩家棋子位置
  - 通道1：对手玩家棋子位置
  - 通道2：对手最近一步落子位置
  - 通道3：当前玩家是否先手（黑棋=全1，白棋=全0）

### 残差塔
- 4个残差块，每个残差块包含：
  ```
  Conv2d(64→64) → BatchNorm → ReLU → Conv2d(64→64) → BatchNorm → Add → ReLU
  ```
- 卷积核大小：3×3，填充：1
- 通道数：64

### 策略头（Policy Head）
- Conv2d(64→2) → BatchNorm → ReLU
- 展平 → Linear(50→26) → LogSoftmax
- 输出：26维对数概率分布（25个棋盘位置 + 1个PASS）

### 价值头（Value Head）
- Conv2d(64→1) → BatchNorm → ReLU
- 展平 → Linear(25→64) → ReLU → Linear(64→1) → Tanh
- 输出：标量价值估计 [-1, 1]

### 损失函数
- **策略损失**：交叉熵损失
- **价值损失**：均方误差损失
- **总损失**：策略损失 + 价值损失

## 训练数据

训练数据格式：`(state, policy, value)`
- state：(4, 5, 5) 棋盘状态（4通道输入）
- policy：(26,) 策略分布（25个棋盘位置 + 1个PASS）
- value：标量价值（1=胜，-1=负，0=平局）

### 训练数据统计
- 训练数据通过自我对弈动态生成，不单独保存
- 总训练局数：约15万局自我对弈数据
- 数据生成时间：2026年3月29日（01:11:19 - 16:46:56）

## 模型文件

训练好的模型保存在以下位置：
- `alpha/model_best.pth`：最佳模型
- `alpha/models/model_iter_XXX.pth`：各轮迭代模型（共15个迭代版本）

### 训练环境
- **硬件**：NVIDIA vGPU-48GB-350W
- **训练时长**：30小时
- **总训练步数**：约45轮（从日志分析）

## 训练曲线

### 训练曲线生成
```bash
python alpha/plot_training_curves.py
```
- 读取 `alpha/training_log.txt` 中的训练数据
- 将每一行视为一轮训练（不考虑Iteration数字）
- 生成Loss和PolicyEntropy的折线图
- 保存为 `alpha/training_curves_new.png`

### 训练统计结果
- **总训练轮数**：45轮
- **初始Loss**：3.8240 → **最终Loss**：1.6358（下降约57%）
- **初始PolicyEntropy**：3.1308 → **最终PolicyEntropy**：1.2349（下降约60%）
- **训练趋势**：Loss和Entropy均呈现明显下降趋势，表明模型逐渐收敛

## 性能优化

1. **CUDA加速**：自动检测GPU并使用
2. **并行自我对弈**：支持多进程并行生成训练数据
3. **缓存优化**：缓存合法落子位置，减少重复计算
4. **批量处理**：训练时使用批量数据提高效率

## 扩展功能

项目支持与多种对手进行对战：
- RandomPlayer：随机落子
- MCTSPlayerV1：基础MCTS玩家
- MCTSPlayerV2：优化版MCTS玩家
- XshPlayer：其他AI玩家
- human_gui_player：人类GUI玩家（修改play_vs_with_host.bat的host为host_gui.py）


## 开发说明

### 添加新玩家
在 `alpha/players/` 目录下创建新的玩家类，实现 `get_input(go, piece_type)` 方法。

### 修改棋盘大小
修改代码中的 `board_size` 参数（默认5），同时更新神经网络的输入输出维度。

### 调整训练参数
修改 `alpha/train.py` 中的训练参数，如：
- `num_games_per_iteration`：每轮自我对弈局数
- `num_simulations`：MCTS模拟次数
- `learning_rate`：学习率
- `batch_size`：批量大小

## 版权声明

本项目为西安交通大学电气工程学院人工智能导论课程项目，仅供学习和研究使用，不用于商业用途，也不承担任何责任。

## 联系方式

- 作者：西安交通大学电气工程及其自动化23级学生xjc
- QQ：3656234737