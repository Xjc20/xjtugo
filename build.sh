#!/bin/bash
# =============================================================================
# 文件功能: 围棋AI比赛自动化测试脚本 (Bash)
#
# 主要功能:
# 1. 自动检测编程语言 (Python/C++/Java)
# 2. 编译对应语言的玩家程序
# 3. 与随机玩家(random_player)进行多轮对战
# 4. 记录比赛结果到cresult.txt
#
# 参数说明:
# - play_time: 比赛轮数 (默认4轮，黑子白子各2轮)
# - ta_agent: 对手列表 (默认random_player)
# - prefix: 路径前缀 (默认"./")
#
# 支持的编程语言:
# - Python: my_player.py
# - C++: my_player.cpp / my_player11.cpp (C++11)
# - Java: my_player.java
#
# 使用方法:
# 1. 确保my_player.xxx文件存在于当前目录
# 2. 运行: bash build.sh
# 3. 查看结果: cat cresult.txt
#
# 比赛流程:
# 1. 清理临时文件
# 2. 检测并编译玩家程序
# 3. 进行多轮对战 (轮流执黑/执白)
# 4. 统计胜负并输出总结
# 5. 清理临时文件
#
# 注意:
# - 需要Python环境运行host.py
# - 需要init/input.txt作为初始棋盘状态
# =============================================================================

# 清理之前编译生成的共享库文件
rm -rf *.so

# ============================================
# 步骤1: 检测编程语言类型
# ============================================
echo "Programming language..."

# 查找当前目录下所有以my_player开头的文件
command=$(ls|grep my_player)

# 使用正则表达式检测不同语言的源文件
# 检测Python文件
py=$([[ $command =~ (^|[[:space:]])"my_player.py"($|[[:space:]]) ]] && echo 'yes' || echo 'no')
py3=$([[ $command =~ (^|[[:space:]])"my_player.py"($|[[:space:]]) ]] && echo 'yes' || echo 'no')
# 检测C++文件
cpp=$([[ $command =~ (^|[[:space:]])"my_player.cpp"($|[[:space:]]) ]] && echo 'yes' || echo 'no')
# 检测C++11文件
c11=$([[ $command =~ (^|[[:space:]])"my_player11.cpp"($|[[:space:]]) ]] && echo 'yes' || echo 'no')
# 检测Java文件
java=$([[ $command =~ (^|[[:space:]])"my_player.java"($|[[:space:]]) ]] && echo 'yes' || echo 'no')

# ============================================
# 步骤2: 根据检测到的语言编译/设置运行命令
# ============================================
if [ "$py" == "yes" ]; then
	# Python版本
	cmd="python my_player.py"
	echo "PY"
elif [ "$py3" == "yes" ]; then
    # Python3版本
    cmd="python my_player.py"
	echo "PY3"
elif [ "$cpp" == "yes" ]; then
	# C++版本: 使用g++编译，优化级别O2
	g++ -O2 my_player.cpp -o exe
	cmd="./exe"
	echo "CPP"
elif [ "$java" == "yes" ]; then
	# Java版本: 先编译再运行
	javac my_player.java
	cmd="java my_player"
	echo "JAVA"
elif [ "$c11" == "yes" ]; then
	# C++11版本: 使用C++0x标准
	g++ -std=c++0x -O2 my_player11.cpp -o exe
	cmd="./exe"
	echo "11"
else
    # 未检测到支持的文件类型
    echo "ERROR: INVALID FILENAME"
	exit 1
fi

echo ""

# ============================================
# 步骤3: 配置比赛参数
# ============================================

# 路径前缀
prefix="./"
# 对手列表 (TA提供的测试玩家)
ta_agent=("random_player") # 1 TA players
# 文件后缀
surfix=".py"

# ============================================
# 函数: play - 进行一局比赛
# 参数: $1 - 黑方命令, $2 - 白方命令
# 返回值: 比赛结果 (0=平局, 1=黑胜, 2=白胜)
# ============================================
play()
{    
    # 清理临时文件
    echo Clean up... >&2
    echo Clean up... >&2 >> cresult.txt
    if [ -f "input.txt" ]; then
        rm input.txt
    fi
    if [ -f "output.txt" ]; then
        rm output.txt
    fi
    # 复制初始棋盘状态
    cp $prefix/init/input.txt ./input.txt

    echo Start Playing... >&2
    echo Start Playing... >&2 >>cresult.txt

    # 初始化移动计数器
	moves=0
	# 游戏主循环
	while true
	do
        # 清理上一步的输出文件
        if [ -f "output.txt" ]; then
	        rm output.txt
	    fi

        # ---------- 黑方回合 ----------
        echo "Black makes move..." >&2
        echo "Black makes move..." >&2 >> cresult.txt
        # 执行黑方AI程序
		eval "$1" >&2
		let moves+=1

        # 调用host.py判断游戏状态和合法性
		python $prefix/host.py -m $moves -v True >&2
		rst=$?  # 获取返回值

        # 如果返回值非0，表示游戏结束
		if [[ "$rst" != "0" ]]; then
			break
		fi

        # 清理黑方的输出文件
        if [ -f "output.txt" ]; then
	        rm output.txt
	    fi

        # ---------- 白方回合 ----------
		echo "White makes move..." >&2
        echo "White makes move..." >&2 >> cresult.txt
        # 执行白方AI程序
		eval "$2" >&2
		let moves+=1

        # 调用host.py判断游戏状态
		python $prefix/host.py -m $moves -v True >&2
		rst=$?

        # 如果返回值非0，表示游戏结束
		if [[ "$rst" != "0" ]]; then
			break
		fi
	done

    # 返回比赛结果
	echo $rst
}

# ============================================
# 步骤4: 设置比赛轮数
# ============================================
# 总比赛轮数 (必须是偶数，轮流执黑/执白)
play_time=4

# ============================================
# 步骤5: 开始比赛主循环
# ============================================

echo ""
echo $(date)  # 输出开始时间

# 遍历所有对手 (这里只有一个random_player)
for i in {0..0} # 1 TA players
do
    echo ""
    echo ==Playing with ${ta_agent[i]}==
    echo $(date)
    # 同时将比赛信息写入结果文件
    echo "" >>cresult.txt
    echo ==Playing with ${ta_agent[i]}== >>cresult.txt
    echo $(date) >>cresult.txt
    
    # 构建对手的完整命令
    ta_cmd="python $prefix${ta_agent[i]}$surfix"
    
    # 初始化胜负统计变量
    black_win_time=0    # 执黑获胜次数
    white_win_time=0    # 执白获胜次数
    black_tie=0         # 执黑平局次数
    white_tie=0         # 执白平局次数
    
    # 比赛轮次循环 (每次循环进行2轮: 对方执黑、我方执黑)
    for (( round=1; round<=$play_time; round+=2 )) 
    do
        # ---------- 第round轮: TA执黑，我方执白 ----------
        echo "=====Round $round====="
        echo "=====Round $round=====" >> cresult.txt
        echo Black:TA White:You 
        echo Black:TA White:You >> cresult.txt
        
        # 调用play函数进行比赛
        # 参数: $ta_cmd (黑方-TA), $cmd (白方-我方)
        winner=$(play "$ta_cmd" "$cmd")
        
        # 判断比赛结果 (2=白方胜, 0=平局, 其他=白方负)
        if [[ "$winner" = "2" ]]; then
            echo 'White(You) win!'
            echo 'White(You) win!'>>cresult.txt
            let white_win_time+=1
        elif [[ "$winner" = "0" ]]; then
            echo Tie.
            let white_tie+=1
        else
            echo 'White(You) lose.'
        fi

        # ---------- 第round+1轮: 我方执黑，TA执白 ----------
        echo "=====Round $((round+1))====="
        echo "=====Round $((round+1))=====" >> cresult.txt
        echo Black:You White:TA
        
        # 调用play函数进行比赛
        # 参数: $cmd (黑方-我方), $ta_cmd (白方-TA)
        winner=$(play "$cmd" "$ta_cmd")
        
        # 判断比赛结果 (1=黑方胜, 0=平局, 其他=黑方负)
        if [[ "$winner" = "1" ]]; then
            echo 'Black(You) win!'
            echo 'Black(You) win!' >> cresult.txt
            let black_win_time+=1
        elif [[ "$winner" = "0" ]]; then
            echo Tie.
            let black_tie+=1
        else
            echo 'Black(You) lose.'
            echo 'Black(You) lose.'>> cresult.txt
        fi
    done

    # ============================================
    # 步骤6: 输出比赛总结
    # ============================================
    echo =====Summary=====  
    echo "You play as Black Player | Win: $black_win_time | Lose: $((play_time/2-black_win_time-black_tie)) | Tie: $black_tie"
    echo "You play as White Player | Win: $white_win_time | Lose: $((play_time/2-white_win_time-black_tie)) | Tie: $white_tie"
    
    # 同时将总结写入结果文件
    echo =====Summary=====  >> cresult.txt
    echo "You play as Black Player | Win: $black_win_time | Lose: $((play_time/2-black_win_time-black_tie)) | Tie: $black_tie" >>cresult.txt
    echo "You play as White Player | Win: $white_win_time | Lose: $((play_time/2-white_win_time-black_tie)) | Tie: $white_tie" >>cresult.txt
done

# ============================================
# 步骤7: 清理临时文件
# ============================================

# 删除input.txt和output.txt
if [ -f "input.txt" ]; then
    rm input.txt
fi
if [ -f "output.txt" ]; then
    rm output.txt
fi

# 删除Java编译生成的class文件                                      
if [ -e "my_player.class" ]; then
    rm *.class
fi

# 删除C/C++编译生成的可执行文件
if [ -e "exe" ]; then
    rm exe
fi

# 删除Python缓存目录
if [ -e "__pycache__" ]; then
    rm -rf __pycache__
fi

# ============================================
# 比赛结束
# ============================================
echo ""
echo Mission Completed.
echo $(date)  # 输出结束时间