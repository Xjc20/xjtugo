import tkinter as tk
from host_gui import GO, GOGUI

# 创建一个简单的围棋游戏实例
go = GO(5)
go.init_board(5)

# 创建 GUI 并运行
gui = GOGUI(go)
gui.create_window()
try:
    gui.root.mainloop()
except:
    pass
