
"""
文件功能: 围棋游戏输出写入模块，负责将AI决策结果写入文件

主要功能:
1. writeOutput函数: 将落子位置或PASS写入output.txt
2. writePass函数: 专门用于写入PASS
3. writeNextInput函数: 生成下一步的input.txt文件内容

参数说明:
- result: 落子位置元组 (row, col) 或字符串 "PASS"
- path: 输出文件路径 (默认"output.txt")
- piece_type: 下一步玩家的棋子类型 (1=黑子, 2=白子)
- previous_board: 当前棋盘状态 (将成为下一步的previous_board)
- board: 落子后的棋盘状态

输出文件格式:
1. output.txt:
   - "PASS" 表示跳过回合
   - "x,y" 表示在(x,y)位置落子 (如 "2,2")

2. input.txt (由writeNextInput生成):
   第1行: piece_type (下一步玩家)
   第2-6行: previous_board (5行)
   第7-11行: board (5行)

使用方法:
    from write import writeOutput, writePass, writeNextInput
    
    # 写入落子位置
    writeOutput((2, 2))  # 输出: "2,2"
    
    # 写入PASS
    writePass()  # 或 writeOutput("PASS")
    
    # 生成下一步输入
    writeNextInput(2, prev_board, curr_board)  # 白子下一步
"""

def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
    	res = "PASS"
    else:
	    res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)

def writePass(path="output.txt"):
	with open(path, 'w') as f:
		f.write("PASS")

def writeNextInput(piece_type, previous_board, board, path="input.txt"):
	res = ""
	res += str(piece_type) + "\n"
	for item in previous_board:
		res += "".join([str(x) for x in item])
		res += "\n"
        
	for item in board:
		res += "".join([str(x) for x in item])
		res += "\n"

	with open(path, 'w') as f:
		f.write(res[:-1])

		