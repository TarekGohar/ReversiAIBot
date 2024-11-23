# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

"""
Helpers.py is a collection of functions that primarily make up the Reversi/Othello game logic.
Beyond a few things in the World init, which can be copy/pasted this should be almost
all of what you'll need to simulate games in your search method.

Functions:
    get_directions    - a simple helper to deal with the geometry of Reversi moves
    count_capture     - how many flips does this move make. Game logic defines valid moves as those with >0 returns from this function. 
    count_capture_dir - a helper for the above, unlikely to be used externally
    execute_move      - update the chess_board by simulating a move
    flip_disks        - a helper for the above, unlikely to be used externally
    check_endgame     - check for termination, who's won but also helpful to score non-terminated games
    get_valid_moves   - use this to get the children in your tree
    random_move       - basis of the random agent and can be used to simulate play

    For all, the chess_board is an np array of integers, size nxn and integer values indicating square occupancies.
    The current player is (1: Blue, 2: Brown), 0's in the board mean empty squares.
    Move pos is a tuple holding [row,col], zero indexed such that valid entries are [0,board_size-1]
"""

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    self.maxMoves = 0

  def calculate_UCB(self, avg, n, N, c=1.414):
    return avg + c * np.sqrt(np.log(N) / n)

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    valid_moves = get_valid_moves(chess_board, player)
    node_scores = np.full(len(valid_moves), np.inf)
    print(node_scores)
    w_l = []


    # print("----------------- Checking Valid Moves -----------------")

    start_time = time.time()
    
    for move in valid_moves:
      new_board = deepcopy(chess_board)
      execute_move(new_board, move, player)
      cur_player = opponent
      iterations = 5
      # print("Move: ", move)
      final_scores = []

      # print("------------ START ---------------")

      for _ in range(iterations):
        # print("-----------------Simulating Game-----------------")
        iteration_board = deepcopy(new_board)
        total_moves = 0
        moves = 0
        # print("-----------------End Simulating Game-----------------")
        while True:
          is_endgame, p0_score, p1_score = check_endgame(iteration_board, player, opponent)
          if is_endgame:
            final_scores.append((p0_score, p1_score, total_moves, 1 if p0_score > p1_score else 0))
            break

          # print(f"------------ {moves} ---------------")
          move = random_move(iteration_board, cur_player)
          if move is None:
            break
          execute_move(iteration_board, move, cur_player)

          if cur_player == player:
            cur_player = opponent
          else:
            cur_player = player

          total_moves += 1
          # print("Total Moves: ", total_moves)
          moves += 1
      
      # print("Final Scores: ", final_scores)
      score_sum = sum(t[-1] for t in final_scores)
      print("Score Sum: ", score_sum)
      w_l.append(score_sum / iterations)

      # print("------------ END ---------------")
      # print("")
    # print("----------------- End Valid Moves -----------------")
    print("")

    print("Win Loss: ", w_l)
    winner_index = w_l.index(max(w_l))
    time_taken = time.time() - start_time

    print("Time Taken: ", time_taken)
    return valid_moves[winner_index]
   

    # print("My AI's turn took ", time_taken, "seconds.")
    # print("MYINFO: ", get_valid_moves(chess_board, player))

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    # return random_move(chess_board,player)

