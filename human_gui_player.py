import sys
import argparse
from host_gui import GO, GOGUI, writeOutput


class HumanGUIPlayer:
    def __init__(self):
        self.type = "human_gui"

    def get_input(self, go, piece_type):
        gui = GOGUI(go)
        gui.current_piece = piece_type
        gui.create_window()
        result = gui.get_input()
        if result == "PASS":
            writeOutput("PASS")
        elif result == "RESTART":
            writeOutput("RESTART")
        elif result:
            writeOutput(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--size", type=int, default=5, help="board size")
    args = parser.parse_args()

    from read import readInput
    from write import writeNextInput

    N = args.size
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)

    player = HumanGUIPlayer()
    action = player.get_input(go, piece_type)
