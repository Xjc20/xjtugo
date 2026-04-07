import random
import math
import time
from read import readInput
from write import writeOutput

BOARD_SIZE = 5
EMPTY, BLACK, WHITE = 0, 1, 2
KOMI = 2.5

# 中心 > 星位 > 边 > 角
POS_SCORE = [
    [0.8, 1.2, 1.5, 1.2, 0.8],
    [1.2, 4.5, 5.0, 4.5, 1.2],
    [1.5, 5.0, 7.5, 5.0, 1.5],
    [1.2, 4.5, 5.0, 4.5, 1.2],
    [0.8, 1.2, 1.5, 1.2, 0.8]
]

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def other(piece):
    return 3 - piece


def fast_copy(board):
    return [row[:] for row in board]


def board_key(board):
    return tuple(tuple(row) for row in board)


def in_bounds(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def count_empty(board):
    return sum(cell == EMPTY for row in board for cell in row)


def game_phase(board):
    """
    依据空点数判断局面阶段：
    1 = 开中盘，2 = 中盘，3 = 后盘
    """
    empties = count_empty(board)
    if empties >= 17:
        return 1
    if empties >= 9:
        return 2
    return 3


def get_group_and_liberties(board, r, c):
    """返回该棋块及其气。"""
    color = board[r][c]
    if color == EMPTY:
        return set(), set()

    group = {(r, c)}
    liberties = set()
    stack = [(r, c)]

    while stack:
        x, y = stack.pop()
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny):
                continue
            if board[nx][ny] == EMPTY:
                liberties.add((nx, ny))
            elif board[nx][ny] == color and (nx, ny) not in group:
                group.add((nx, ny))
                stack.append((nx, ny))

    return group, liberties


def remove_dead_groups(board, color_to_check):
    """扫描整盘，清除某一颜色所有无气棋块。"""
    visited = set()
    removed = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] != color_to_check or (i, j) in visited:
                continue
            group, liberties = get_group_and_liberties(board, i, j)
            visited |= group
            if not liberties:
                removed += len(group)
                for x, y in group:
                    board[x][y] = EMPTY
    return removed


def apply_move(board, prev_board, move, piece):
    """
    在 board 上落子，返回:
    (是否合法, 新棋盘, 新的 previous_board, 提子数)
    """
    if move is None:
        return True, fast_copy(board), fast_copy(board), 0

    r, c = move
    if not in_bounds(r, c) or board[r][c] != EMPTY:
        return False, None, None, 0

    new_board = fast_copy(board)
    new_board[r][c] = piece

    captured = remove_dead_groups(new_board, other(piece))

    # 再检查己方是否自杀
    _, liberties = get_group_and_liberties(new_board, r, c)
    if not liberties:
        return False, None, None, 0

    # 简单打劫：新盘面不能和上一手盘面相同
    if prev_board is not None and board_key(new_board) == board_key(prev_board):
        return False, None, None, 0

    return True, new_board, fast_copy(board), captured


def territory_estimate(board, root_piece):
    """极简空点围地估计。"""
    visited = set()
    score = 0

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] != EMPTY or (i, j) in visited:
                continue

            region = set()
            borders = set()
            stack = [(i, j)]
            visited.add((i, j))
            region.add((i, j))

            while stack:
                x, y = stack.pop()
                for dx, dy in DIRS:
                    nx, ny = x + dx, y + dy
                    if not in_bounds(nx, ny):
                        continue
                    if board[nx][ny] == EMPTY and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        region.add((nx, ny))
                        stack.append((nx, ny))
                    elif board[nx][ny] != EMPTY:
                        borders.add(board[nx][ny])

            if len(borders) == 1:
                owner = next(iter(borders))
                if owner == root_piece:
                    score += len(region)
                else:
                    score -= len(region)

    return score


def evaluate_board(board, root_piece):
    """
    终局/截断评估：返回 root_piece 视角下 [0,1] 的胜率估计。
    这里改成分阶段权重，避免早中期被 territory 误导。
    """
    stone_diff = 0
    liberty_diff = 0
    pos_diff = 0.0
    atari_diff = 0

    seen = set()
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == EMPTY or (i, j) in seen:
                continue

            group, liberties = get_group_and_liberties(board, i, j)
            seen |= group

            group_size = len(group)
            group_pos = sum(POS_SCORE[x][y] for x, y in group)
            group_lib = len(liberties)

            if board[i][j] == root_piece:
                stone_diff += group_size
                liberty_diff += group_lib
                pos_diff += group_pos
                if group_lib == 1:
                    atari_diff -= group_size
            else:
                stone_diff -= group_size
                liberty_diff -= group_lib
                pos_diff -= group_pos
                if group_lib == 1:
                    atari_diff += group_size

    terr_diff = territory_estimate(board, root_piece)
    empties = count_empty(board)

    # 分阶段权重：前期看形，后期看地
    if empties >= 17:
        stone_w = 2.10
        terr_w = 0.55
        liberty_w = 0.45
        pos_w = 0.12
        atari_w = 0.25
        scale = 2.8
    elif empties >= 9:
        stone_w = 2.20
        terr_w = 0.95
        liberty_w = 0.30
        pos_w = 0.10
        atari_w = 0.35
        scale = 2.5
    else:
        stone_w = 2.30
        terr_w = 1.30
        liberty_w = 0.18
        pos_w = 0.06
        atari_w = 0.45
        scale = 2.2

    diff = (
        stone_w * stone_diff
        + terr_w * terr_diff
        + liberty_w * liberty_diff
        + pos_w * pos_diff
        + atari_w * atari_diff
    )

    if root_piece == WHITE:
        diff += KOMI
    else:
        diff -= KOMI

    x = max(min(diff / scale, 15.0), -15.0)
    return 1.0 / (1.0 + math.exp(-x))


class Node:
    __slots__ = (
        "board", "prev_board", "player", "move", "parent",
        "children", "visits", "wins"
    )

    def __init__(self, board, prev_board, player, move=None, parent=None):
        self.board = board
        self.prev_board = prev_board
        self.player = player
        self.move = move
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.wins = 0.0  # 统一存“root 视角”的胜率


class MCTS:
    def __init__(self, time_limit=8.5):
        self.time_limit = time_limit
        self.legal_cache = {}
        self.random = random.Random()
        self.root_piece = BLACK

    def expansion_limit(self, board, node_visits):
        """
        保留 A 的框架，但不要一上来就把所有低价值招都拖进树里。
        访问越多，允许扩展的候选越多。
        """
        empties = count_empty(board)
        if empties >= 17:
            base = 8
        elif empties >= 9:
            base = 9
        else:
            base = 10

        # 节点越热，允许稍微扩宽
        bonus = min(4, node_visits // 40)
        return min(BOARD_SIZE * BOARD_SIZE + 1, base + bonus)

    def rollout_topk(self, board):
        empties = count_empty(board)
        if empties >= 17:
            return 3
        if empties >= 9:
            return 4
        return 5

    def fallback_move(self, board, prev_board, piece):
        """
        统一兜底：返回一个可下的真实坐标。
        这个函数只在极端情况下调用，用来避免 writeOutput 收到 None。
        """
        moves = self.legal_moves(board, prev_board, piece)
        real_moves = [m for m in moves if m is not None]
        if real_moves:
            return max(real_moves, key=lambda m: self.move_priority(board, prev_board, m, piece))

        # 理论上不太会走到这里；再兜一层，保证不是 None
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == EMPTY:
                    return (i, j)

        return (2, 2)

    def legal_moves(self, board, prev_board, piece):
        """
        生成合法着法（含 PASS），并按启发式排序。
        缓存 key 必须包含 prev_board，否则打劫判断会错。
        """
        key = (board_key(board), board_key(prev_board) if prev_board is not None else None, piece)
        if key in self.legal_cache:
            return self.legal_cache[key][:]

        moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] != EMPTY:
                    continue
                ok, _, _, _ = apply_move(board, prev_board, (i, j), piece)
                if ok:
                    moves.append((i, j))

        # PASS 永远可下；但开局不宜过早 pass，所以用低优先级保留在列表里
        moves.append(None)
        moves.sort(key=lambda m: self.move_priority(board, prev_board, m, piece), reverse=True)
        self.legal_cache[key] = moves[:]
        return moves

    def move_priority(self, board, prev_board, move, piece):
        """
        用于排序和 rollout 选招的启发式分数，越大越优。
        这里强化吃子、救子、避免自紧气，同时保留原 A 的稳定风格。
        """
        if move is None:
            occupied = BOARD_SIZE * BOARD_SIZE - count_empty(board)
            # 越到后盘，PASS 越可以接受
            phase = game_phase(board)
            if phase == 1:
                return -9.0 + occupied * 0.18
            elif phase == 2:
                return -6.5 + occupied * 0.24
            else:
                return -3.5 + occupied * 0.30

        r, c = move
        score = POS_SCORE[r][c] * 0.60
        opp = other(piece)
        own = piece

        seen_opp = set()
        seen_own = set()
        tactical_capture = 0

        for dx, dy in DIRS:
            nr, ny = r + dx, c + dy
            if not in_bounds(nr, ny):
                continue

            # 吃子/打吃优先
            if board[nr][ny] == opp and (nr, ny) not in seen_opp:
                group, liberties = get_group_and_liberties(board, nr, ny)
                seen_opp |= group
                if (r, c) in liberties:
                    if len(liberties) == 1:
                        score += 6.5 + 2.0 * len(group)
                    elif len(liberties) == 2:
                        score += 2.0 + 0.7 * len(group)
                    elif len(liberties) == 3:
                        score += 0.8 + 0.25 * len(group)

            # 补棋救己方打吃块
            elif board[nr][ny] == own and (nr, ny) not in seen_own:
                group, liberties = get_group_and_liberties(board, nr, ny)
                seen_own |= group
                if (r, c) in liberties and len(liberties) == 1:
                    score += 4.8 + 1.0 * len(group)

        ok, new_board, _, captured = apply_move(board, prev_board, move, piece)
        if not ok:
            return -1e9

        tactical_capture = captured
        _, libs_after = get_group_and_liberties(new_board, r, c)

        # 提子直接加分，尤其是大块
        if tactical_capture > 0:
            score += 1.8 * tactical_capture + 0.6 * min(tactical_capture, 4)

        # 自紧气惩罚
        if len(libs_after) == 1:
            score -= 2.0
        elif len(libs_after) == 0:
            score -= 7.0
        elif len(libs_after) == 2:
            score -= 0.3

        # 更偏向连接成形，但不要过强
        own_neighbors = 0
        for dx, dy in DIRS:
            nr, ny = r + dx, c + dy
            if in_bounds(nr, ny) and board[nr][ny] == own:
                own_neighbors += 1
        if own_neighbors >= 2:
            score += 0.5
        elif own_neighbors == 1:
            score += 0.15

        # 早期更重视中心，中后期也保留一点
        phase = game_phase(board)
        if phase == 1:
            score += POS_SCORE[r][c] * 0.10
        elif phase == 2:
            score += POS_SCORE[r][c] * 0.06
        else:
            score += POS_SCORE[r][c] * 0.04

        # 避免纯粹填自己的真眼
        if tactical_capture == 0:
            surrounded = True
            for dx, dy in DIRS:
                nr, ny = r + dx, c + dy
                if in_bounds(nr, ny) and board[nr][ny] != own:
                    surrounded = False
                    break
            if surrounded:
                score -= 1.6

        return score

    def best_child(self, node):
        """
        wins/visits 存的是 root 视角。
        当前轮到 node.player 下棋时：
          - 如果 node.player == root_piece，选让 root 更有利的子节点
          - 否则选让 root 更不利的子节点（即最小化 root 胜率）
        """
        best_move = None
        best_node = None
        best_value = -1e18

        log_n = math.log(node.visits + 1.0)
        phase = game_phase(node.board)

        # 前期更强调探索，后期更强调利用
        if phase == 1:
            c = 1.35
            prior_scale = 0.035
        elif phase == 2:
            c = 1.10
            prior_scale = 0.030
        else:
            c = 0.90
            prior_scale = 0.025

        maximizing = (node.player == self.root_piece)

        for move, child in node.children.items():
            q = child.wins / child.visits if child.visits > 0 else 0.5
            root_value = q if maximizing else (1.0 - q)
            u = c * math.sqrt(log_n / (child.visits + 1.0))
            prior = self.move_priority(node.board, node.prev_board, move, node.player) * prior_scale
            value = root_value + u + prior
            if value > best_value:
                best_value = value
                best_move = move
                best_node = child

        return best_move, best_node

    def select_and_expand(self, root):
        """
        从 root 出发，选择/扩展到一个新节点。
        返回 (节点, 该节点局面board, 该节点局面prev_board, 轮到谁下)
        """
        node = root
        board = fast_copy(root.board)
        prev_board = fast_copy(root.prev_board) if root.prev_board is not None else None
        player = root.player

        while True:
            moves = self.legal_moves(board, prev_board, player)
            if not moves:
                return node, board, prev_board, player

            # 只在当前阶段允许的“候选宽度”内找未扩展的招法
            limit = self.expansion_limit(board, node.visits)
            expand_pool = moves[:min(limit, len(moves))]
            untried = [m for m in expand_pool if m not in node.children]

            if untried:
                move = untried[0]
                ok, new_board, new_prev, _ = apply_move(board, prev_board, move, player)
                if not ok:
                    return node, board, prev_board, player
                child = Node(new_board, new_prev, other(player), move, node)
                node.children[move] = child
                return child, new_board, new_prev, other(player)

            move, child = self.best_child(node)
            if child is None:
                return node, board, prev_board, player

            node = child
            board = fast_copy(node.board)
            prev_board = fast_copy(node.prev_board) if node.prev_board is not None else None
            player = node.player

    def rollout(self, board, prev_board, player, root_piece):
        """
        轻量 rollout：保留随机性，但让不同阶段的走法偏好不同。
        """
        sim_board = fast_copy(board)
        sim_prev = fast_copy(prev_board) if prev_board is not None else None
        turn = player
        consecutive_pass = 0

        phase = game_phase(sim_board)
        max_steps = 20 if phase == 1 else 24 if phase == 2 else 28

        for _ in range(max_steps):
            moves = self.legal_moves(sim_board, sim_prev, turn)
            if not moves:
                break

            empties = count_empty(sim_board)
            topk = min(self.rollout_topk(sim_board), len(moves))
            ranked = moves[:topk]

            # 前期尽量少随机，后期稍微放开一点
            if empties >= 17:
                move = ranked[0]
            elif empties >= 9:
                if len(ranked) >= 3 and self.random.random() < 0.25:
                    move = self.random.choice(ranked[:3])
                else:
                    move = ranked[0]
            else:
                if len(ranked) >= 4 and self.random.random() < 0.35:
                    move = self.random.choice(ranked[:4])
                else:
                    move = ranked[0]

            # 空点很多时不主动 pass
            if move is None and empties > 8:
                move = ranked[0] if ranked[0] is not None else None

            if move is None:
                consecutive_pass += 1
                if consecutive_pass >= 2:
                    break
                sim_prev = fast_copy(sim_board)
                turn = other(turn)
                continue
            else:
                consecutive_pass = 0

            ok, new_board, new_prev, _ = apply_move(sim_board, sim_prev, move, turn)
            if not ok:
                # 兜底：如果启发式选到非法局面，临时让其 pass
                consecutive_pass += 1
                if consecutive_pass >= 2:
                    break
                sim_prev = fast_copy(sim_board)
                turn = other(turn)
                continue

            sim_board = new_board
            sim_prev = new_prev
            turn = other(turn)

        return evaluate_board(sim_board, root_piece)

    def backpropagate(self, node, value):
        """value 统一是 root 视角胜率，向上直接回传即可。"""
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.wins += value
            cur = cur.parent

    def opening_move(self, board, prev_board, piece):
        """
        小棋盘开局书：优先中心，其次星位/近星位。
        只在非常空的局面启用。
        """
        if count_empty(board) < 21:
            return None

        candidates = [
            (2, 2),
            (1, 1), (1, 3), (3, 1), (3, 3),
            (1, 2), (2, 1), (2, 3), (3, 2),
            (0, 0), (0, 4), (4, 0), (4, 4)
        ]

        best_move = None
        best_score = -1e18
        for move in candidates:
            ok, _, _, _ = apply_move(board, prev_board, move, piece)
            if not ok:
                continue
            s = self.move_priority(board, prev_board, move, piece)
            if s > best_score:
                best_score = s
                best_move = move

        return best_move

    def run(self, board, prev_board, piece):
        self.root_piece = piece
        self.legal_cache.clear()

        opening = self.opening_move(board, prev_board, piece)
        if opening is not None:
            return opening

        root = Node(fast_copy(board), fast_copy(prev_board) if prev_board is not None else None, piece)
        deadline = time.monotonic() + self.time_limit

        while time.monotonic() < deadline:
            node, sim_board, sim_prev, sim_player = self.select_and_expand(root)
            value = self.rollout(sim_board, sim_prev, sim_player, piece)
            self.backpropagate(node, value)

        if not root.children:
            # 这里不再返回 "PASS"，而是返回一个可写出的真实坐标
            return self.fallback_move(board, prev_board, piece)

        # root 的子节点是“我走一步后”的局面，root 视角下 wins 越大越好
        best_move = max(
            root.children.items(),
            key=lambda kv: (
                kv[1].visits,
                kv[1].wins / kv[1].visits if kv[1].visits > 0 else 0.0
            )
        )[0]

        # 保险兜底，避免 writeOutput 收到 None
        if best_move is None:
            return self.fallback_move(board, prev_board, piece)

        return best_move


if __name__ == "__main__":
    p_type, prev_b, curr_b = readInput(BOARD_SIZE)
    solver = MCTS(time_limit=8.5)
    action = solver.run(curr_b, prev_b, p_type)

    # 额外兜底，确保不会把 None 传给 writeOutput
    if not (isinstance(action, tuple) and len(action) == 2):
        action = solver.fallback_move(curr_b, prev_b, p_type)

    writeOutput(action)