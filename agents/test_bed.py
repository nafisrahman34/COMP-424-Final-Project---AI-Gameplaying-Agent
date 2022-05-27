import copy
import numpy as np
from random import choices
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
    # root_node = Node(chess_board, my_pos, adv_pos, 0, 0, 0, -1000, 0)  # score is low so we don't bother exploring this node
    valid_positions = set()  # set of 2-tuples storing coordinates of valid positions. format: (vert cord, horz coord)
    queue = [root_node.my_pos]
    while queue:
        cell = queue.pop(0)  # current cell, a 2-tuple in form: (vertical coord, horizontal coord). same as (row #, column #)
        valid_positions.add(cell)  # staying still is valid
        manhattan_distance = abs(root_node.my_pos[0] - cell[0]) + abs(root_node.my_pos[1] - cell[1])
        if manhattan_distance >= this_max_step:
            break  # Need to break once we're k steps from root

        for dir1 in range(4):
            if not root_node.chess_board[cell[0], cell[1], dir1]:  # checks if no barrier is there (also checks for borders as a result)

                if dir1 == 0 and (cell[0] - 1, cell[1]) not in valid_positions and root_node.adv_pos != (
                        cell[0] - 1, cell[1]):  # move up; don't move into other player
                    queue.append((cell[0] - 1, cell[1]))
                    valid_positions.add((cell[0] - 1, cell[1]))

                elif dir1 == 1 and (cell[0], cell[1] + 1) not in valid_positions and root_node.adv_pos != (
                        cell[0], cell[1] + 1):  # move right; don't move into other player
                    queue.append((cell[0], cell[1] + 1))
                    valid_positions.add((cell[0], cell[1] + 1))

                elif dir1 == 2 and (cell[0] + 1, cell[1]) not in valid_positions and root_node.adv_pos != (
                        cell[0] + 1, cell[1]):  # move down; don't move into other player
                    queue.append((cell[0] + 1, cell[1]))
                    valid_positions.add((cell[0] + 1, cell[1]))

                elif dir1 == 3 and (cell[0], cell[1] - 1) not in valid_positions and root_node.adv_pos != (
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
# Index in dim2 represents [Up, Right, Down, Left] respectively
# Record barriers and boarders for each block; making 6x6 chessboard
board_size = 6

a_chess_board = np.zeros((board_size, board_size, 4), dtype=bool)

# Set borders
a_chess_board[0, :, 0] = True  # sets top barrier by making the corresponding values in the 3-D array "1"
a_chess_board[:, 0, 3] = True  # same for left barrier
a_chess_board[-1, :, 2] = True  # -1 here means access the last element in that list
a_chess_board[:, -1, 1] = True

# Maximum Steps
max_step = (board_size + 1) // 2

opposites = {0: 2, 1: 3, 2: 0, 3: 1}
moves = ((-1, 0), (0, 1), (1, 0), (0, -1))


def set_barrier(this_chess_board, row, col, dir1):
    # Set the barrier to True
    this_chess_board[row, col, dir1] = True
    # Set the opposite barrier to True
    move = moves[dir1]
    this_chess_board[row + move[0], col + move[1], opposites[dir1]] = True


# Random barriers (symmetric)
for _ in range(max_step):
    pos = np.random.randint(0, board_size, size=2)
    r, c = pos
    direction = np.random.randint(0, 4)
    while a_chess_board[r, c, direction]:
        pos = np.random.randint(0, board_size, size=2)
        r, c = pos
        direction = np.random.randint(0, 4)
    anti_pos = board_size - 1 - pos
    anti_dir = opposites[direction]
    anti_r, anti_c = anti_pos
    set_barrier(a_chess_board, r, c, direction)
    set_barrier(a_chess_board, anti_r, anti_c, anti_dir)

# Random start position (symmetric but not overlap)
main_player_pos = np.random.randint(0, board_size, size=2)
adv_player_pos = board_size - 1 - main_player_pos
while np.array_equal(main_player_pos, adv_player_pos):
    main_player_pos = np.random.randint(0, board_size, size=2)
    adv_player_pos = board_size - 1 - main_player_pos
main_player_pos = tuple(main_player_pos)
adv_player_pos = tuple(adv_player_pos)


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



"""

----------- TESTING BLOCK ---------------------------------------------------------------------------
"""
a_chess_board[2, 0, 0] = True
a_chess_board[2, 0, 1] = True
main_player_pos = (2, 1)
adv_player_pos = (0, 2)
test_node = Node(0, a_chess_board, main_player_pos, adv_player_pos, 8, 4, 1000, None)
valid_moves = find_student_agent_possible_moves(test_node, max_step)
invalid_moves = []
for i in range(len(valid_moves)):
    start_pos = np.asarray(main_player_pos)
    end_pos = np.asarray(valid_moves[i].my_pos)
    bool = check_valid_step(test_node.chess_board, start_pos, end_pos, adv_player_pos, valid_moves[i].barrier_dir, max_step)
    if not bool:
        print("flag of dicks failed")
        invalid_moves.append()
#bool = check_valid_step(a_chess_board, start_pos, end_pos, adv_player_pos, valid_moves[0].barrier_dir, max_step)
#print (str(bool) + " lksdjflkdsjf lk ")
"""
for i in range(len(valid_moves)):
    start_pos = np.asarray(main_player_pos)
    end_pos = np.asarray(valid_moves[i].my_pos)
    bool = check_valid_step(test_node.chess_board, start_pos, end_pos, adv_player_pos, valid_moves[i].barrier_dir, max_step)
    if not bool:
        print("flag of dicks failed")

counter = 0
for i in range(len(valid_moves)):
    a_child_valid_moves = find_student_agent_possible_moves(valid_moves[i], max_step)
    start_pos1 = np.asarray(valid_moves[i].my_pos)
    for j in range(len(a_child_valid_moves)):
        end_pos1 = np.asarray(a_child_valid_moves[j].my_pos)
        bool1 = check_valid_step(valid_moves[i].chess_board, start_pos1, end_pos1, adv_player_pos, a_child_valid_moves[j].barrier_dir, max_step)
        if not bool1:
            print("flag failed")
            counter += 1

print(len(valid_moves))
print(counter/(len(valid_moves)))
"""
# test root node
# print(a_chess_board)
# print(main_player_pos)
# print("\n\n" + "MARKER")
"""
start_time = time.time_ns()
test_valid_moves = find_student_agent_possible_moves(test_node, max_step)
print(str(len(test_valid_moves)))
best_score = 0
best_node = None
for i in range(len(test_valid_moves)):

    for j in range(5):
        simulate(test_valid_moves[i], max_step)
        j = j+1
    update_upwards(test_valid_moves[i], 5)

    #numerator = test_valid_moves[i].n_times_won
    #denominator = test_valid_moves[i].n_times_simulated
    #print("numerator:"+str(numerator)+" denominator:"+str(denominator))
    test_valid_moves[i].score = calculate_mcts_score(test_valid_moves[i])
    score = test_valid_moves[i].score
    if score>best_score:
        best_score = score
        best_node = test_valid_moves[i]
    #print("score of node "+str(i)+": "+str(score))

runtime = time.time_ns() - start_time
print(str(runtime))

test_valid_moves[0].n_times_simulated = 3
test_valid_moves[0].n_times_won = 1
some_score = calculate_mcts_score(test_valid_moves[0])
print("dicks " + str(some_score))
result = math.log(120)
print(str(result) + " fffff")
"""

"""
test_valid_moves = find_student_agent_possible_moves(test_node, max_step)
result = simulate(test_valid_moves[0], max_step)  # 1 for win, 0 for loss/tie
update_upwards(test_valid_moves[0])
print("RESULT: " + str(result))
print("Root_node times simulated: " + str(test_node.n_times_simulated))
print("Root_node times won: " + str(test_node.n_times_won))
print("child's parent times simulated: " + str(test_valid_moves[0].n_parent_simulations))
"""

"""
a_time = time.time_ns()

#test_valid_moves = find_student_agent_possible_moves(test_node, max_step)
#for i in range(len(test_valid_moves)):
#   some_shit = find_student_agent_possible_moves(test_valid_moves[i], max_step)
b_time = time.time_ns()

print(b_time-a_time)
print(b_time)
print(a_time)
#print("amount of children: " + str(len(test_node.children)))
#total_time = 0
"""
""""
for i in range(1000):
    start_time = time.time_ns()
    test_valid_moves = find_student_agent_possible_moves(test_node, max_step)
    run_time = time.time_ns() - start_time
    total_time += run_time
#end_time = time.time_ns()
print(total_time)
"""
""""
for i in range(100):
    random_student_agent_walk(a_chess_board, test_node.my_pos, test_node.adv_pos, max_step)
    the_result = check_endgame(a_chess_board, test_node.my_pos, test_node.adv_pos)

"""
#print("IT IS : " + str(the_result[0]))

#print("RESULT: " + str(result))
# find_possible_moves(test_node, max_step)

# print(test_node)

#print("length of list: " + str(len(test_valid_moves)))
#print(test_valid_moves)
# print("amount of children: " + str(len(test_node.children)))
#print(str(start_time) + " , " + str(end_time))
