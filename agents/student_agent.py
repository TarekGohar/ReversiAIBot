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

  def calculate_UCB(self, w, n, N, c=np.sqrt(2)):
    if n == 0:
      return np.inf

    value = w/n + c * np.sqrt(np.log(N) / n)
    return value

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

    start_time = time.time()
    AVG_ITERATIONS = 3

    total_iterations = 0
    valid_moves = get_valid_moves(chess_board, player)
    move_scores = {move: 0 for move in valid_moves}
    move_counts = {move: 0 for move in valid_moves}
    move_UCB = {move: self.calculate_UCB(move_scores[move], move_counts[move], total_iterations) for move in valid_moves}

    while total_iterations < len(valid_moves) * AVG_ITERATIONS:
      best_move = max(move_UCB, key=move_UCB.get)
      new_board = deepcopy(chess_board)
      execute_move(new_board, best_move, player)
      cur_user = opponent
      potential_stalemate = False
      while True:
        is_endgame, p0_score, p1_score = check_endgame(new_board, player, opponent)
        if is_endgame:
          if move_scores[best_move] == np.inf:
            move_scores[best_move] = 1 if p0_score > p1_score else 0
          else:
            move_scores[best_move] += 1 if p0_score > p1_score else 0

          move_counts[best_move] += 1
          total_iterations += 1
          move_UCB[best_move] = self.calculate_UCB(move_scores[best_move], move_counts[best_move], total_iterations)
          break

        move = random_move(new_board, cur_user)
        if move is None:
          print("       No valid moves")
          if potential_stalemate:
            if move_scores[best_move] == np.inf:
              move_scores[best_move] = 0

            move_counts[best_move] += 1
            total_iterations += 1
            move_UCB[best_move] = self.calculate_UCB(move_scores[best_move], move_counts[best_move], total_iterations)
            break

          potential_stalemate = True

          if cur_user == player:
            cur_user = opponent
          else:
            cur_user = player
          continue

        execute_move(new_board, move, cur_user)

        if cur_user == player:
          cur_user = opponent
        else:
          cur_user = player

    print("       Final Choice", max(move_UCB, key=move_UCB.get))
    return max(move_UCB, key=move_UCB.get)

