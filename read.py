"""
文件功能: 围棋游戏输入读取模块，负责从文件读取游戏状态

主要功能:
1. readInput函数: 读取input.txt文件，解析当前玩家、上一步棋盘和当前棋盘
2. readOutput函数: 读取output.txt文件，解析对手的落子位置

参数说明:
- n: 棋盘大小 (5x5棋盘则n=5)
- path: 输入文件路径 (默认"input.txt")
- piece_type: 当前玩家棋子类型 (1=黑子, 2=白子)
- previous_board: 上一步棋盘状态 (二维列表)
- board: 当前棋盘状态 (二维列表)

输入文件格式(input.txt):
第1行: piece_type (1或2)
第2-6行: previous_board (5行，每行5个数字)
第7-11行: board (5行，每行5个数字)

输出文件格式(output.txt):
- "PASS" 表示跳过
- "x,y" 表示落子位置 (如 "2,2")

返回值:
- readInput: (piece_type, previous_board, board)
- readOutput: (action, x, y) action为"MOVE"或"PASS"

使用方法:
    from read import readInput, readOutput
    piece_type, prev_board, curr_board = readInput(5)
    action, x, y = readOutput()
"""

def readInput(n, path="input.txt"):

    with open(path, "r") as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [
            [int(x) for x in line.rstrip("\n")] for line in lines[1 : n + 1]
        ]
        board = [
            [int(x) for x in line.rstrip("\n")] for line in lines[n + 1 : 2 * n + 1]
        ]

        return piece_type, previous_board, board


def readOutput(path="output.txt"):
    with open(path, "r") as f:
        position = f.readline().strip().split(",")

        if position[0] == "PASS":
            return "PASS", -1, -1

        x = int(position[0])
        y = int(position[1])

    return "MOVE", x, y
