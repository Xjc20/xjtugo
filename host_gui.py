import sys
import random
import timeit
import math
import argparse
import tkinter as tk
from tkinter import messagebox
from collections import Counter
from copy import deepcopy

from read import *
from write import writeNextInput, writeOutput


path = "./cresult.txt"


class GO:
    def __init__(self, n):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        self.previous_board = None
        self.X_move = True
        self.died_pieces = []
        self.n_move = 0
        self.max_move = n * n - 1
        self.komi = n / 2
        self.verbose = False

    def init_board(self, n):
        """
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        """
        board = [[0 for x in range(n)] for y in range(n)]
        self.board = board
        self.previous_board = deepcopy(board)

    def set_board(self, piece_type, previous_board, board):
        """
        Initialize board status.
        :param previous_board: previous board state.
        :param board: current board state.
        :return: None.
        """

        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        self.previous_board = previous_board
        self.board = board

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        """
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        """
        return deepcopy(self)

    def detect_neighbor(self, i, j):
        """
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        """
        board = self.board
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, j))
        if i < len(board) - 1:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < len(board) - 1:
            neighbors.append((i, j + 1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        """
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        """
        board = self.board
        neighbors = self.detect_neighbor(i, j)
        group_allies = []
        for piece in neighbors:
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        """
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        """
        stack = [(i, j)]
        ally_members = []
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        """
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        """
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                if board[piece[0]][piece[1]] == 0:
                    return True
        return False

    def find_died_pieces(self, piece_type):
        """
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        """
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == piece_type:
                    if not self.find_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        """
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        """

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces:
            return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        """
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        """
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def place_chess(self, i, j, piece_type):
        """
        Place a chess stone in the board.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        """
        board = self.board

        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update_board(board)
        return True

    def valid_place_check(self, i, j, piece_type, test_check=False):
        """
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        """

        f = open(path, "a")
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

        if not (i >= 0 and i < len(board)):
            if verbose:
                print(
                    ("Invalid placement. row should be in the range 1 to {}.").format(
                        len(board) - 1
                    )
                )
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(
                    (
                        "Invalid placement. column should be in the range 1 to {}."
                    ).format(len(board) - 1)
                )
            return False

        if board[i][j] != 0:
            if verbose:
                print("Invalid placement. There is already a chess in this position.")
            return False

        test_go = self.copy_board()
        test_board = test_go.board

        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            if verbose:
                print("Invalid placement. No liberty found in this position.")
                print("Invalid placement. No liberty found in this position.", file=f)
            return False

        else:
            if self.died_pieces and self.compare_board(
                self.previous_board, test_go.board
            ):
                if verbose:
                    print(
                        "Invalid placement. A repeat move not permitted by the KO rule."
                    )
                    print(
                        "Invalid placement. A repeat move not permitted by the KO rule.",
                        file=f,
                    )
                return False
        return True

    def update_board(self, new_board):
        """
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        """
        self.board = new_board

    def visualize_board(self):
        """
        Visualize the board.

        :return: None
        """
        board = self.board

        print("-" * len(board) * 2)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 0:
                    print(" ", end=" ")
                elif board[i][j] == 1:
                    print("X", end=" ")
                else:
                    print("O", end=" ")
            print()
        print("-" * len(board) * 2)

        f = open(path, "a")
        print("-" * len(board) * 2, file=f)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 0:
                    print(" ", end=" ", file=f)
                elif board[i][j] == 1:
                    print("X", end=" ", file=f)
                else:
                    print("O", end=" ", file=f)
            print(file=f)
        print("-" * len(board) * 2, file=f)
        f.close()

    def game_end(self, piece_type, action="MOVE"):
        """
        Check if the game should end.

        :param piece_type: 1('X') or 2('O').
        :param action: "MOVE" or "PASS".
        :return: boolean indicating whether the game should end.
        """

        if self.n_move >= self.max_move:
            return True
        if self.compare_board(self.previous_board, self.board) and action == "PASS":
            return True
        return False

    def score(self, piece_type):
        """
        Get score of a player by counting the number of stones.

        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the game should end.
        """

        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt

    def judge_winner(self):
        """
        Judge the winner of the game by number of pieces for each player.

        :param: None.
        :return: piece type of winner of the game (0 if it's a tie).
        """

        cnt_1 = self.score(1)
        cnt_2 = self.score(2)
        if cnt_1 > cnt_2 + self.komi:
            return 1
        elif cnt_1 < cnt_2 + self.komi:
            return 2
        else:
            return 0

    def play(self, player1, player2, verbose=False):
        """
        The game starts!

        :param player1: Player instance.
        :param player2: Player instance.
        :param verbose: whether print input hint and error information
        :return: piece type of winner of the game (0 if it's a tie).
        """

        f = open(path, "a")
        self.init_board(self.size)
        if player1.type == "manual" or player2.type == "manual":
            self.verbose = True
            print('----------Input "exit" to exit the program----------')
            print("X stands for black chess, O stands for white chess.")
            self.visualize_board()

        verbose = self.verbose
        while 1:
            piece_type = 1 if self.X_move else 2

            if self.game_end(piece_type):
                result = self.judge_winner()
                if verbose:
                    print("Game ended.")
                    print("Game ended", file=f)
                    if result == 0:
                        print("The game is a tie.")
                    else:
                        print(
                            "The winner is {}, black score {} white score {}".format(
                                " X " if result == 1 else " O ",
                                self.score(1),
                                self.score(2),
                            )
                        )
                        print(
                            "The winner is {}, black score {} white score {}".format(
                                " X " if result == 1 else " O ",
                                self.score(1),
                                self.score(2),
                            ),
                            file=f,
                        )
                return result

            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(player + " makes move...")
                print(player + " makes move...", file=f)

            if piece_type == 1:
                action = player1.get_input(self, piece_type)
            else:
                action = player2.get_input(self, piece_type)

            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(action)

            if action != "PASS":
                if not self.place_chess(action[0], action[1], piece_type):
                    if verbose:
                        self.visualize_board()
                    continue

                self.died_pieces = self.remove_died_pieces(
                    3 - piece_type
                )
            else:
                self.previous_board = deepcopy(self.board)

            if verbose:
                self.visualize_board()
                print()
                print(file=f)

            self.n_move += 1
            self.X_move = not self.X_move
            f.close()


class GOGUI:
    def __init__(self, go, board_size=500):
        self.go = go
        self.board_size = board_size
        self.cell_size = board_size // go.size
        self.padding = max(self.cell_size // 2 + 5, 40)
        self.canvas_size = self.board_size + self.padding * 2
        self.root = None
        self.canvas = None
        self.buttons = []
        self.current_piece = 1
        self.legal_moves = set()
        self.selected = {"move": None}
        self.game_over = False
        self.update_legal_moves()

    def update_legal_moves(self):
        self.legal_moves = set()
        for i in range(self.go.size):
            for j in range(self.go.size):
                if self.go.valid_place_check(i, j, self.current_piece, test_check=True):
                    self.legal_moves.add((i, j))

    def create_window(self):
        self.root = tk.Tk()
        self.root.title("围棋游戏 - Go Game")
        self.root.resizable(False, False)
        self.root.configure(bg="#2C2C2C")
        self.root.update_idletasks()
        
        # 计算足够宽的窗口尺寸
        board_total_size = self.canvas_size + 16
        win_w = board_total_size + 100
        win_h = board_total_size + 200
        
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - win_w) // 2
        y = (sh - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        # 信息区域
        info_frame = tk.Frame(self.root, bg="#2C2C2C")
        info_frame.pack(pady=(10, 5))

        self.info_label = tk.Label(
            info_frame,
            text=f"当前执子: {'黑子 ●' if self.current_piece == 1 else '白子 ○'}",
            font=("Microsoft YaHei", 16, "bold"),
            fg="#FFFFFF",
            bg="#2C2C2C",
        )
        self.info_label.pack()

        self.status_label = tk.Label(
            info_frame,
            text=f"步数: {self.go.n_move}    棋盘: {self.go.size}×{self.go.size}",
            font=("Microsoft YaHei", 11),
            fg="#AAAAAA",
            bg="#2C2C2C",
        )
        self.status_label.pack()

        # 棋盘区域 - 居中
        board_frame = tk.Frame(self.root, bg="#2C2C2C")
        board_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        canvas_bg_frame = tk.Frame(board_frame, bg="#8B7355", padx=8, pady=8)
        canvas_bg_frame.pack(anchor=tk.CENTER)

        self.canvas = tk.Canvas(
            canvas_bg_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#DEB887",
            highlightthickness=0,
        )
        self.canvas.pack()

        self.draw_board()
        self.draw_stones()

        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # 按钮区域
        btn_frame = tk.Frame(self.root, bg="#2C2C2C")
        btn_frame.pack(pady=10)

        self.pass_btn = tk.Button(
            btn_frame,
            text="PASS 停一手",
            width=14,
            height=2,
            font=("Microsoft YaHei", 13),
            bg="#4A4A4A",
            fg="#FFFFFF",
            activebackground="#6A6A6A",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            command=self.on_pass,
        )
        self.pass_btn.pack(side=tk.LEFT, padx=20)

        self.restart_btn = tk.Button(
            btn_frame,
            text="重新开始",
            width=14,
            height=2,
            font=("Microsoft YaHei", 13),
            bg="#4A4A4A",
            fg="#FFFFFF",
            activebackground="#6A6A6A",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            command=self.on_restart,
        )
        self.restart_btn.pack(side=tk.LEFT, padx=20)

    def draw_board(self):
        self.canvas.delete("grid")
        padding = self.padding
        last_line = padding + (self.go.size - 1) * self.cell_size
        for i in range(self.go.size):
            x = padding + i * self.cell_size
            self.canvas.create_line(
                x, padding, x, last_line, fill="#5D4D37", tags="grid", width=2
            )
            self.canvas.create_line(
                padding, padding + i * self.cell_size,
                last_line, padding + i * self.cell_size,
                fill="#5D4D37", tags="grid", width=2
            )

        star_points = []
        if self.go.size == 5:
            star_points = [(2, 2)]
        elif self.go.size >= 7:
            offsets = [2, self.go.size // 2, self.go.size - 3]
            for ox in offsets:
                for oy in offsets:
                    star_points.append((ox, oy))
        for (i, j) in star_points:
            x = padding + j * self.cell_size
            y = padding + i * self.cell_size
            self.canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5, fill="#5D4D37", tags="grid"
            )

        for i in range(self.go.size):
            x = padding + i * self.cell_size
            self.canvas.create_text(
                x, padding - 20, text=str(i), font=("Arial", 11, "bold"), fill="#5D4D37", tags="grid"
            )
            self.canvas.create_text(
                padding - 20, padding + i * self.cell_size,
                text=str(i), font=("Arial", 11, "bold"), fill="#5D4D37", tags="grid"
            )

    def draw_stones(self):
        self.canvas.delete("stone")
        padding = self.padding
        radius = self.cell_size // 2 - 4
        for i in range(self.go.size):
            for j in range(self.go.size):
                stone = self.go.board[i][j]
                if stone != 0:
                    x = padding + j * self.cell_size
                    y = padding + i * self.cell_size
                    color = "black" if stone == 1 else "white"
                    outline = "#333333" if stone == 2 else "#999999"
                    self.canvas.create_oval(
                        x - radius, y - radius, x + radius, y + radius,
                        fill=color, outline=outline, width=2, tags="stone"
                    )
                    if stone == 2:
                        self.canvas.create_oval(
                            x - radius + 3, y - radius + 3,
                            x + radius - 3, y + radius - 3,
                            fill="", outline="#DDDDDD", width=1, tags="stone"
                        )

        for (i, j) in self.legal_moves:
            x = padding + j * self.cell_size
            y = padding + i * self.cell_size
            self.canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4,
                fill="#228B22", outline="#145214", tags="stone"
            )

    def on_canvas_click(self, event):
        if self.game_over:
            return
        padding = self.padding
        col = int((event.x - padding + self.cell_size // 2) // self.cell_size)
        row = int((event.y - padding + self.cell_size // 2) // self.cell_size)

        if 0 <= row < self.go.size and 0 <= col < self.go.size:
            if (row, col) in self.legal_moves:
                self.selected["move"] = (row, col)
                self.root.quit()

    def on_pass(self):
        if self.game_over:
            return
        self.selected["move"] = "PASS"
        self.root.quit()

    def on_restart(self):
        self.selected["move"] = "RESTART"
        self.root.quit()

    def show_game_over(self, winner, black_score, white_score):
        self.game_over = True
        if winner == 0:
            msg = f"游戏结束！\n平局！\n黑子: {black_score} 子\n白子: {white_score} 子\n贴目: {self.go.komi}"
        elif winner == 1:
            msg = f"游戏结束！\n黑子胜！\n黑子: {black_score} 子\n白子: {white_score} 子\n贴目: {self.go.komi}"
        else:
            msg = f"游戏结束！\n白子胜！\n黑子: {black_score} 子\n白子: {white_score} 子\n贴目: {self.go.komi}"
        messagebox.showinfo("游戏结束", msg)

    def get_input(self):
        self.selected["move"] = None

        if self.root is None:
            self.create_window()
        else:
            self.info_label.config(
                text=f"当前执子: {'黑子 ●' if self.current_piece == 1 else '白子 ○'}"
            )
            self.status_label.config(text=f"步数: {self.go.n_move}    棋盘: {self.go.size}×{self.go.size}")
            self.draw_stones()

        self.root.mainloop()

        result = self.selected["move"]
        self.root.destroy()
        self.root = None
        self.canvas = None

        return result

    def refresh(self):
        if self.root:
            self.update_legal_moves()
            self.info_label.config(
                text=f"当前执子: {'黑子 ●' if self.current_piece == 1 else '白子 ○'}"
            )
            self.status_label.config(text=f"步数: {self.go.n_move}    棋盘: {self.go.size}×{self.go.size}")
            self.draw_board()
            self.draw_stones()


class HumanGUIPlayer:
    def __init__(self):
        self.type = "human_gui"

    def get_input(self, go, piece_type):
        gui = GOGUI(go)
        gui.current_piece = piece_type
        return gui.get_input()


def judge(n_move, verbose=False):

    f = open(path, "a")

    N = 5

    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.verbose = verbose
    go.set_board(piece_type, previous_board, board)
    go.n_move = n_move
    try:
        action, x, y = readOutput()
    except:
        print("output.txt not found or invalid format")
        print("output.txt not found or invalid format", file=f)
        sys.exit(3 - piece_type)

    if action == "MOVE":
        if not go.place_chess(x, y, piece_type):
            print("Game end.")
            print("The winner is {}".format("X" if 3 - piece_type == 1 else "O"))
            print("Game end.", file=f)
            print(
                "The winner is {}".format("X" if 3 - piece_type == 1 else "O"), file=f
            )
            sys.exit(3 - piece_type)

        go.died_pieces = go.remove_died_pieces(3 - piece_type)

    if verbose:
        go.visualize_board()
        print()
        print(file=f)

    if go.game_end(piece_type, action):
        result = go.judge_winner()
        if verbose:
            print("Game end.")
            print("Game end.", file=f)
            if result == 0:
                print("The game is a tie.")
                print("The game is a tie.", file=f)
            else:
                print(
                    "The winner is {}, black score {} white score {}".format(
                        "X" if result == 1 else "O", go.score(1), go.score(2)
                    )
                )
                print(
                    "The winner is {}, black score {} white score {}".format(
                        "X" if result == 1 else "O", go.score(1), go.score(2)
                    ),
                    file=f,
                )
        sys.exit(result)

    piece_type = 2 if piece_type == 1 else 1

    if action == "PASS":
        go.previous_board = go.board
    writeNextInput(piece_type, go.previous_board, go.board)
    f.close()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--move", "-m", type=int, help="number of total moves", default=0
    )
    parser.add_argument("--verbose", "-v", type=bool, help="print board", default=False)
    args = parser.parse_args()

    judge(args.move, args.verbose)
