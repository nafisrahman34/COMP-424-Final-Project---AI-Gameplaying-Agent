# Student agent: Add your own agent here
import sys
import time

from agents.game_tree import *
from agents.agent import Agent
from store import register_agent
from random import choices
import numpy as np


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        # This is a dictionary (Key: Value pair)
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.time_limit = 2
        self.turn_counter = 1
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        start_time = time.time()
        if self.turn_counter == 1:
            self.time_limit = 29
        else:
            self.time_limit = 1.9
        root_node = Node(0, chess_board, my_pos, adv_pos, 0, 0, -100, None)
        valid_next_moves = find_student_agent_possible_moves(root_node, max_step)  # step 1 from google doc

        good_moves = choices(valid_next_moves, k=15)
        some_valid_moves = []
        for a in range(15):
            start_pos = np.asarray(my_pos)
            end_pos = np.asarray(good_moves[a].my_pos)
            if check_valid_step(chess_board, start_pos, end_pos, adv_pos, good_moves[a].barrier_dir, max_step):
                some_valid_moves.append(good_moves[a])

        break_outer = False
        best_node = some_valid_moves[0]
        best_score = 0
        while True:
            for i in range(len(some_valid_moves)):  # step 2 from google doc
                remaining_time = self.time_limit - (time.time() - start_time)
                for j in range(2):
                    break_inner = False
                    simulate_result = simulate(some_valid_moves[i], max_step, remaining_time)
                    if simulate_result == -1:
                        break_outer = True
                        break_inner = True
                        break
                if break_inner:
                    break
                update_upwards(some_valid_moves[i], 2)
                some_valid_moves[i].score = calculate_mcts_score(some_valid_moves[i])
                if some_valid_moves[i].score > best_score:
                    best_score = some_valid_moves[i].score
                    best_node = some_valid_moves[i]

            if break_outer:
                break
        
        self.turn_counter += 1
        return best_node.my_pos, best_node.barrier_dir
