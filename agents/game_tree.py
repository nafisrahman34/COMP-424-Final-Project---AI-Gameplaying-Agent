import copy
import time
import numpy as np
import math


class Node:
    # Constructor: int, np array M*M*4, 2-tuple, 2-tuple, int, int, float, depth
    def __init__(self, depth, chess_board, my_pos, adv_pos, n_times_simulated, n_times_won, score, barrier_dir):
        self.children = []  # this sets the attributes
        self.parent = None
        self.depth = depth # we pick the node with greatest score at depth = 1 (b/c that's the next move); depth(root) = 0
        self.chess_board = chess_board  # numpy boolean array that is MXMX4, showing the barriers
        self.my_pos = my_pos  # tuple of (vertical, horizontal)
        self.adv_pos = adv_pos
        self.n_times_simulated = n_times_simulated      # num of times this node and its children were simulated
        self.n_times_won = n_times_won                   # num of times this node won (includes num of times children won)
        self.score = score                                # UCB1/UCT score calculated w/special equation
        self.barrier_dir = barrier_dir

    def add_child(self, child):
        child.depth = self.depth + 1  # child's depth is 1 more than parent's
        child.n_parent_simulations = self.n_times_simulated
        child.parent = self
        self.children.append(child)

    def __repr__(self):
        string = str(self.my_pos) + "\n " + str(self.chess_board) + "\n\n\n"
        return string
        #return str(self.children)


# returns list of possible moves. in form: [Node1, Node2, etc] & adds children to the "root_node"
# To be called when root_node.depth is odd
def find_student_agent_possible_moves(root_node, this_max_step):
    valid_positions = set()  # set of 2-tuples storing coordinates of valid positions. format: (vert cord, horz coord)
    queue = [root_node.my_pos]
    while queue:
        cell = queue.pop(0)  # 2-tuple in form: (vertical coord, horizontal coord). same as (row #, column #)
        valid_positions.add(cell)  # staying still is valid
        manhattan_distance = abs(root_node.my_pos[0] - cell[0]) + abs(root_node.my_pos[1] - cell[1])
        if manhattan_distance >= this_max_step:
            break  # Need to break once we're k steps from root

        for i in range(4):
            if not root_node.chess_board[cell[0], cell[1], i]:  # checks if no barrier is there (also checks for borders as a result)

                if i == 0 and (cell[0] - 1, cell[1]) not in valid_positions and root_node.adv_pos != (
                        cell[0] - 1, cell[1]):  # move up; don't move into other player
                    queue.append((cell[0] - 1, cell[1]))
                    valid_positions.add((cell[0] - 1, cell[1]))

                elif i == 1 and (cell[0], cell[1] + 1) not in valid_positions and root_node.adv_pos != (
                        cell[0], cell[1] + 1):  # move right; don't move into other player
                    queue.append((cell[0], cell[1] + 1))
                    valid_positions.add((cell[0], cell[1] + 1))

                elif i == 2 and (cell[0] + 1, cell[1]) not in valid_positions and root_node.adv_pos != (
                        cell[0] + 1, cell[1]):  # move down; don't move into other player
                    queue.append((cell[0] + 1, cell[1]))
                    valid_positions.add((cell[0] + 1, cell[1]))

                elif i == 3 and (cell[0], cell[1] - 1) not in valid_positions and root_node.adv_pos != (
                        cell[0], cell[1] - 1):  # move left; don't move into other player
                    queue.append((cell[0], cell[1] - 1))
                    valid_positions.add((cell[0], cell[1] - 1))

    possible_moves = []  # list of possible moves; each move is a node in the game tree
    # loop fills possible_moves w/ all possible moves by including every barrier option.
    for position in valid_positions:
        for j in range(4):
            if not root_node.chess_board[position[0], position[1], j]:
                new_board = copy.deepcopy(root_node.chess_board)
                set_barrier(new_board, position[0], position[1], j)
               # new_board[position[0], position[1], j] = True  # set the border to true
               # move = moves[j]
               # new_board[position[0] + move[0], position[1] + move[1], opposites[j]] = True
                node = Node(root_node.depth + 1, new_board, position, root_node.adv_pos, 0, 0, 1000, j)  # setting new node to have large value score
                possible_moves.append(node)
                root_node.add_child(node)

    return possible_moves
# returns list of possible moves. in form: [Node1, Node2, etc] & adds children to the "root_node"
# To be called when root_node.depth is even
def find_opponent_possible_moves(root_node, this_max_step):
    valid_positions = set()  # set of 2-tuples storing coordinates of valid positions. format: (vert cord, horz coord)
    queue = [root_node.adv_pos]
    while queue:
        cell = queue.pop(0)  # 2-tuple in form: (vertical coord, horizontal coord). same as (row #, column #)
        valid_positions.add(cell)  # staying still is valid
        manhattan_distance = abs(root_node.adv_pos[0] - cell[0]) + abs(root_node.adv_pos[1] - cell[1])
        if manhattan_distance >= this_max_step:
            break  # Need to break once we're k steps from root

        for i in range(4):
            if not root_node.chess_board[cell[0], cell[1], i]:  # checks if no barrier is there (also checks for borders as a result)

                if i == 0 and (cell[0] - 1, cell[1]) not in valid_positions and root_node.my_pos != (
                        cell[0] - 1, cell[1]):  # move up; don't move into other player
                    queue.append((cell[0] - 1, cell[1]))
                    valid_positions.add((cell[0] - 1, cell[1]))

                elif i == 1 and (cell[0], cell[1] + 1) not in valid_positions and root_node.my_pos != (
                        cell[0], cell[1] + 1):  # move right; don't move into other player
                    queue.append((cell[0], cell[1] + 1))
                    valid_positions.add((cell[0], cell[1] + 1))

                elif i == 2 and (cell[0] + 1, cell[1]) not in valid_positions and root_node.my_pos != (
                        cell[0] + 1, cell[1]):  # move down; don't move into other player
                    queue.append((cell[0] + 1, cell[1]))
                    valid_positions.add((cell[0] + 1, cell[1]))

                elif i == 3 and (cell[0], cell[1] - 1) not in valid_positions and root_node.my_pos != (
                        cell[0], cell[1] - 1):  # move left; don't move into other player
                    queue.append((cell[0], cell[1] - 1))
                    valid_positions.add((cell[0], cell[1] - 1))

    possible_moves = []  # list of possible moves; each move is a node in the game tree
    # loop fills possible_moves w/ all possible moves by including every barrier option.
    for position in valid_positions:
        for j in range(4):
            if not root_node.chess_board[position[0], position[1], j]:
                new_board = copy.deepcopy(root_node.chess_board)
                set_barrier(new_board, position[0], position[1], j)
              #  new_board[position[0], position[1], j] = True  # set the border to true
               # move = moves[j]
               # new_board[position[0] + move[0], position[1] + move[1], opposites[j]] = True
                node = Node(root_node.depth + 1, new_board, root_node.my_pos, position, 0, 0, 1000, j)  # setting new node to have large value score
                possible_moves.append(node)
                root_node.add_child(node)

    return possible_moves
# returns my_pos (main player's position in (row #, column #) ) and direction of barrier they place
def random_student_agent_walk(chess_board, my_pos, adv_pos, this_max_step):
        """
        Randomly walk to the next position in the board.

        Parameters
        ----------
        chess_board: M*M*4 numpy array of board
            Position of barriers (indicated by "True")
        my_pos : tuple
            The position of the agent.
        adv_pos : tuple
            The position of the adversary.
        this_max_step: integer
            Max step length K = ceil(M/2)
        """
        ori_pos = copy.deepcopy(my_pos)
        steps = np.random.randint(0, this_max_step + 1)
        # Random Walk
        for _ in range(steps):
            row, col = my_pos
            dir1 = np.random.randint(0, 4)
            m_r, m_c = moves[dir1]
            my_pos = (row + m_r, col + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[row, col, dir1] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir1 = np.random.randint(0, 4)
                m_r, m_c = moves[dir1]
                my_pos = (row + m_r, col + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir1 = np.random.randint(0, 4)
        row, col = my_pos
        while chess_board[row, col, dir1]:
            dir1 = np.random.randint(0, 4)

        return my_pos, dir1

# returns adv_pos (main player's position in (row #, column #) ) and direction of barrier they place
def random_opponent_walk(chess_board, my_pos, adv_pos, this_max_step):
        """
        Randomly walk to the next position in the board.

        Parameters
        ----------
        my_pos : tuple
            The position of the agent.
        adv_pos : tuple
            The position of the adversary.
        chess_board : numpy M*M*4 array
            state of the chess board (barrier positions)
        this_max_step : integer
            max K number of steps (ceiling of M/2)
        """
        ori_pos = copy.deepcopy(adv_pos)
        steps = np.random.randint(0, this_max_step + 1)
        # Random Walk
        for _ in range(steps):
            row, col = adv_pos
            dir1 = np.random.randint(0, 4)
            m_r, m_c = moves[dir1]
            adv_pos = (row + m_r, col + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[row, col, dir1] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir1 = np.random.randint(0, 4)
                m_r, m_c = moves[dir1]
                adv_pos = (row + m_r, col + m_c)

            if k > 300:
                adv_pos = ori_pos
                break

        # Put Barrier
        dir1 = np.random.randint(0, 4)
        row, col = adv_pos
        while chess_board[row, col, dir1]:
            dir1 = np.random.randint(0, 4)

        return adv_pos, dir1


# returns true if game over (false otherwise); also returns player scores
def check_endgame(chess_board, my_pos, adv_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score (student agent) : int
            The score of player 1.
        player_2_score (Opponent) : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        for row in range(len(chess_board)): # len(chess_board) is equivalent to length of the board
            for col in range(len(chess_board)):
                father[(row, col)] = (row, col)

        def find(position):
            if father[position] != position:
                father[position] = find(father[position])
            return father[position]

        def union(pos1, pos2):
            father[pos1] = pos2

        for row in range(len(chess_board)):
            for col in range(len(chess_board)):
                for dir1, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if chess_board[row, col, dir1 + 1]:
                        continue
                    pos_a = find((row, col))
                    pos_b = find((row + move[0], col + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for row in range(len(chess_board)):
            for col in range(len(chess_board)):
                find((row, col))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
        elif p0_score < p1_score:
            player_win = 1
        else:
            player_win = -1  # Tie

        return True, p0_score, p1_score
# NEED TO ADD PROPOGATING UP THE SCORE FROM LEAVES
# returns 1 for student_agent win, 0 for tie or loss
def simulate(node, this_max_step, remaining_time):
    local_start_time = time.time()

    leaf_reached = False
    copy_node = copy.deepcopy(node)
    while not leaf_reached:
        if (time.time()-local_start_time) > (remaining_time - 0.2): # break if 0.1 s left
            return -1
        this_result = check_endgame(copy_node.chess_board, copy_node.my_pos, copy_node.adv_pos)
        if this_result[0]:
            student_agent_score = this_result[1]
            opponent_score = this_result[2]
            leaf_reached = True
            node.n_times_simulated += 1
            if student_agent_score > opponent_score:
                node.n_times_won += 1
             #   return 1
            return

        # opponent agent's moves
        if copy_node.depth % 2 == 0:
            next_step = random_opponent_walk(copy_node.chess_board, copy_node.my_pos, copy_node.adv_pos, this_max_step)
            copy_node.adv_pos = next_step[0]
            adv_x_pos = copy_node.adv_pos[0]
            adv_y_pos = copy_node.adv_pos[1]
            barrier_dir = next_step[1]
            set_barrier(copy_node.chess_board, adv_x_pos, adv_y_pos, barrier_dir)
            copy_node.depth += 1
        # student agent's moves
        else:
            next_step = random_student_agent_walk(copy_node.chess_board, copy_node.my_pos, copy_node.adv_pos, this_max_step)
            copy_node.my_pos = next_step[0]
            my_x_pos = copy_node.my_pos[0]
            my_y_pos = copy_node.my_pos[1]
            student_barrier_dir = next_step[1]
            set_barrier(copy_node.chess_board, my_x_pos, my_y_pos, student_barrier_dir)
            copy_node.depth += 1


# updates parent node’s number of wins and number of simulations (propagates upwards to update all direct ancestors)
# If you want to sim a node multiple times, add times_simulated to parameters
def update_upwards(youngest_child, times_simulated):
    node = youngest_child
    while node.depth > 0:
        node.parent.n_times_won += youngest_child.n_times_won
        node.parent.n_times_simulated += times_simulated  # change to times_simulated
        node.n_parent_simulations += times_simulated      # changes to times_simulated param
        node = node.parent

# using natural log and c = root 2 for now
def calculate_mcts_score(node):
    this_node = node
    exploit_term = this_node.n_times_won / this_node.n_times_simulated
    explore_term = 1.41421 * math.sqrt(math.log(this_node.parent.n_times_simulated) / this_node.n_times_simulated)
    return exploit_term + explore_term


opposites = {0: 2, 1: 3, 2: 0, 3: 1}
moves = ((-1, 0), (0, 1), (1, 0), (0, -1))


def set_barrier(this_chess_board, row, col, dir1):
    # Set the barrier to True
    this_chess_board[row, col, dir1] = True
    # Set the opposite barrier to True
    move = moves[dir1]
    this_chess_board[row + move[0], col + move[1], opposites[dir1]] = True

def check_valid_step(this_chess_board, start_pos, end_pos, adv_pos, barrier_dir, this_max_step):
    """
    Check if the step the agent takes is valid (reachable and within max steps).

    Parameters
    ----------
    start_pos : tuple
        The start position of the agent.
    end_pos : np.ndarray
        The end position of the agent.
    barrier_dir : int
        The direction of the barrier.
    """
    # Endpoint already has barrier or is boarder
    row, col = end_pos
    if this_chess_board[row, col, barrier_dir]:
        return False
    if np.array_equal(start_pos, end_pos):
        return True

    state_queue = [(start_pos, 0)]
    visited = {tuple(start_pos)}
    is_reached = False
    while state_queue and not is_reached:
        cur_pos, cur_step = state_queue.pop(0)
        row, col = cur_pos
        if cur_step == this_max_step:
            break
        for dir1, move in enumerate(moves):
            if this_chess_board[row, col, dir1]:
                continue

            next_pos = cur_pos + move
            if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                continue
            if np.array_equal(next_pos, end_pos):
                is_reached = True
                break

            visited.add(tuple(next_pos))
            state_queue.append((next_pos, cur_step + 1))

    return is_reached

