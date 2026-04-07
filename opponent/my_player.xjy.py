import random
import math
import time
from read import readInput
from write import writeOutput
from host import GO

# ========================
# 1. 权重：中心 > 星位 > 边 > 角
# ========================
POS_SCORE = [
    [0.8, 1.2, 1.5, 1.2, 0.8],
    [1.2, 4.5, 5.0, 4.5, 1.2],
    [1.5, 5.0, 7.5, 5.0, 1.5],
    [1.2, 4.5, 5.0, 4.5, 1.2],
    [0.8, 1.2, 1.5, 1.2, 0.8]
]


def fast_copy(board):
    return [row[:] for row in board]


# ========================
# 2. 围棋逻辑：真眼与气
# ========================
def get_liberties(board, i, j):
    piece = board[i][j]
    visited = {(i, j)}
    stack = [(i, j)]
    libs = set()
    while stack:
        x, y = stack.pop()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 5 and 0 <= ny < 5:
                if board[nx][ny] == 0:
                    libs.add((nx, ny))
                elif board[nx][ny] == piece and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
    return libs


def is_true_eye(board, r, c, piece):
    if board[r][c] != 0: return False
    count = 0
    own = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 5 and 0 <= nc < 5:
            count += 1
            if board[nr][nc] == piece: own += 1
    return count == own


# ========================
# 3. MCTS 终极体
# ========================
class Node:
    __slots__ = ['board', 'prev_board', 'player', 'move', 'parent', 'children', 'visits', 'wins']

    def __init__(self, board, prev_board, player, move=None, parent=None):
        self.board = board
        self.prev_board = prev_board
        self.player = player
        self.move = move
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.wins = 0.0


class MCTS:
    def __init__(self, time_limit=8.7):
        self.time_limit = time_limit
        self.cache = {}

    def get_moves(self, go, piece):
        state_key = (tuple(map(tuple, go.board)), piece)
        if state_key in self.cache: return self.cache[state_key]

        moves = []
        curr_b = fast_copy(go.board)
        prev_b_ref = fast_copy(go.previous_board)

        for i in range(5):
            for j in range(5):
                if is_true_eye(curr_b, i, j, piece): continue
                if go.valid_place_check(i, j, piece, True):
                    # 严格打劫检查
                    go.board = fast_copy(curr_b)
                    go.previous_board = fast_copy(prev_b_ref)

                    go.previous_board = fast_copy(go.board)
                    go.board[i][j] = piece
                    go.remove_died_pieces(3 - piece)

                    if not go.compare_board(go.board, prev_b_ref):
                        moves.append((i, j))

                    go.board = fast_copy(curr_b)

        # 排序：优先尝试高价值点
        moves.sort(key=lambda m: POS_SCORE[m[0]][m[1]], reverse=True)
        self.cache[state_key] = moves
        return moves

    def run(self, go, piece):
        # 1. 开局库
        flat = [c for r in go.board for c in r]
        if flat.count(0) >= 24:
            if go.board[2][2] == 0: return (2, 2)
            if piece == 2: return (1, 1)

        root = Node(fast_copy(go.board), fast_copy(go.previous_board), piece)
        start_time = time.time()

        # 2. 迭代搜索
        while time.time() - start_time < self.time_limit:
            node, sim_go = self.select_and_expand(root, piece)
            reward = self.smart_simulate(sim_go, piece)
            self.backpropagate(node, reward)

        if not root.children: return "PASS"

        # 决策：访问量最大，其次胜率最高
        best_move = max(root.children.items(), key=lambda x: (x[1].visits, x[1].wins / x[1].visits))[0]
        return best_move

    def select_and_expand(self, node, root_p):
        sim_go = GO(5)
        sim_go.set_board(root_p, fast_copy(node.prev_board), fast_copy(node.board))
        curr = node
        curr_p = node.player

        while True:
            moves = self.get_moves(sim_go, curr_p)
            if not moves: return curr, sim_go

            untried = [m for m in moves if m not in curr.children]
            if untried:
                move = untried[0]
                sim_go.previous_board = fast_copy(sim_go.board)
                sim_go.place_chess(move[0], move[1], curr_p)
                sim_go.remove_died_pieces(3 - curr_p)

                new_node = Node(fast_copy(sim_go.board), fast_copy(sim_go.previous_board), 3 - curr_p, move, curr)
                curr.children[move] = new_node
                return new_node, sim_go
            else:
                move, curr = self.best_child(curr)
                sim_go.previous_board = fast_copy(sim_go.board)
                sim_go.place_chess(move[0], move[1], curr_p)
                sim_go.remove_died_pieces(3 - curr_p)
                curr_p = 3 - curr_p

    def best_child(self, node):
        best_val = -1e9
        best_m, best_n = None, None
        # 修正：降低 C 值提高收敛速度
        c = 0.65 if node.visits > 500 else 1.0
        log_v = math.log(node.visits + 1)
        for m, child in node.children.items():
            # PUCT 评分
            score = (child.wins / child.visits) + c * math.sqrt(log_v / child.visits) + (POS_SCORE[m[0]][m[1]] * 0.01)
            if score > best_val:
                best_val, best_m, best_n = score, m, child
        return best_m, best_n

    def smart_simulate(self, go, me):
        """强化模拟：加入贴目感知"""
        p = 1 if go.X_move else 2
        for _ in range(16):
            moves = self.get_moves(go, p)
            if not moves: break

            # 策略：吃子 > 高分位
            best_m = None
            for m in moves:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = m[0] + dx, m[1] + dy
                    if 0 <= nx < 5 and 0 <= ny < 5 and go.board[nx][ny] == 3 - p:
                        if len(get_liberties(go.board, nx, ny)) == 1:
                            best_m = m;
                            break
                if best_m: break

            move = best_m if best_m else random.choice(moves[:min(2, len(moves))])
            go.previous_board = fast_copy(go.board)
            go.place_chess(move[0], move[1], p)
            go.remove_died_pieces(3 - p)
            p = 3 - p

        # 核心修正：5x5 贴目补偿（假设白棋贴 2.5 目）
        s_me = go.score(me)
        s_opp = go.score(3 - me)
        # 如果我是白棋(2)，得分加 2.5；如果是黑棋(1)，对手得分加 2.5
        if me == 2:
            s_me += 2.5
        else:
            s_opp += 2.5

        return 1 if s_me > s_opp else 0

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.wins += reward
            node = node.parent


if __name__ == "__main__":
    p_type, prev_b, curr_b = readInput(5)
    go_obj = GO(5)
    go_obj.set_board(p_type, prev_b, curr_b)

    solver = MCTS(time_limit=8.8)
    action = solver.run(go_obj, p_type)
    writeOutput(action)