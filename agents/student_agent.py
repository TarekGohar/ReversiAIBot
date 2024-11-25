# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves
import math
import psutil
import os


def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")

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

  def step(self, chess_board, player, opponent):
    node = MCTSNode(chess_board, player)
    move = node.mcts_search(chess_board, player)
    # Call periodically in your program
    monitor_memory()
    if move is None:
        print("Player passes.")
    else:
        print(f"Player plays at position {move}.")
    return move
    
# Monte Carlo Tree Search Node class    
class MCTSNode:
    def __init__(self, board_state, current_player, parent=None, move=None):
        self.board_state = board_state # 2D numpy array representing the game
        self.current_player = current_player # Integer representing the current player (1 for Player 1/Blue, 2 for Player 2/Brown)
        self.parent = parent # Parent node
        self.move = move # Move that led to this node
        self.children = [] # List of child nodes
        self.visits = 0 # Number of visits
        self.wins = 0 # Number of wins
        self.rave_visits = {} # Number of RAVE visits for each move
        self.rave_wins = {} # Number of RAVE wins for each move

    def evaluate_board(self, board, color, player_score, opponent_score):
        """
        Evaluate the board state based on multiple factors.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).
        - player_score: Score of the current player.
        - opponent_score: Score of the opponent.

        Returns:
        - int: The evaluated score of the board.
        """
        # Corner positions are highly valuable
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        corner_score = sum(1 for corner in corners if board[corner] == color) * 10
        corner_penalty = sum(1 for corner in corners if board[corner] == 3 - color) * -10

        # Mobility: the number of moves the opponent can make
        opponent_moves = len(get_valid_moves(board, 3 - color))
        mobility_score = -opponent_moves

        # Combine scores
        total_score = player_score - opponent_score + corner_score + corner_penalty + mobility_score
        return total_score

    def mcts_search(self, root_state, player, time_limit=1.9):
      start_time = time.time()
      root_node = MCTSNode(root_state, player)
      iteration = 0
      while time.time() - start_time < time_limit:
          iteration += 1
        #   print("Iteration: ", iteration, "RAM: ", get_ram_usage())
          node = root_node
          # Selection
          is_endgame, _, _ = check_endgame(node.board_state, player, player%2 +1)
          while node.is_fully_expanded() and not is_endgame:
              node = node.best_child()
          # Expansion
          if not is_endgame:
              node.expand()
              if len(node.children) > 0:
                  node = node.children[np.random.randint(len(node.children))]
              else:
                  # No moves to expand, proceed to backpropagation
                  winner = node.get_winner()
                  moves_played = []
                  node.backpropagate(winner, moves_played)
                  continue
          # Simulation
          winner, moves_played = node.rollout()
          # Backpropagation
          node.backpropagate(winner, moves_played)
      # Return the move with the most visits
      if root_node.children:
          best_move = max(root_node.children, key=lambda c: c.visits).move
      else:
          # If no children were expanded, return a random valid move
          valid_moves = get_valid_moves(root_state, player)
          if valid_moves:
              best_move = random.choice(valid_moves)
          else:
              best_move = None  # No valid moves, must pass
      return best_move


    def expand(self):
      if not self.children:
          moves = get_valid_moves(self.board_state, self.current_player)
          if not moves:
              # No valid moves, so the player must pass
              moves = [None]  # Represent pass move with None
          for move in moves:
              new_board = deepcopy(self.board_state)
              if move is not None:
                  execute_move(new_board, move, self.current_player)
              # Switch to the next player
              next_player = self.current_player % 2 + 1
              child_node = MCTSNode(new_board, next_player, parent=self, move=move)
              self.children.append(child_node)


    def is_fully_expanded(self):
        return len(self.children) > 0

    def best_child(self, exploration=1.52):
      best_score = -np.inf
      best_child = None
      for child in self.children:
          if child.visits == 0:
              return child
          rave_visit = child.rave_visits.get(child.move, 0)
          beta = rave_visit / (rave_visit + child.visits + 1e-4)
          q_value = child.wins / child.visits
          rave_win = child.rave_wins.get(child.move, 0)
          rave_q = rave_win / rave_visit if rave_visit > 0 else 0
          uct_value = (1 - beta) * q_value + beta * rave_q + exploration * math.sqrt(math.log(self.visits) / child.visits)
          if uct_value > best_score:
              best_score = uct_value
              best_child = child
      return best_child


    def rollout_policy(self, board_state, player):
      if self.current_player == player:
          return random_move(board_state, player)
      else:
        legal_moves = get_valid_moves(board_state, player)

        if not legal_moves:
            return None  # No valid moves available, pass turn

        # Advanced heuristic: prioritize corners and maximize flips while minimizing opponent's potential moves
        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            simulated_board = deepcopy(board_state)
            execute_move(simulated_board, move, player)
            _, player_score, opponent_score = check_endgame(simulated_board, player, 3 - player)
            move_score = self.evaluate_board(simulated_board, player, player_score, opponent_score)

            if move_score > best_score:
                best_score = move_score
                best_move = move

        # Return the best move found
        return best_move if best_move else random.choice(legal_moves)

    def rollout(self):
      current_state = deepcopy(self.board_state)
      moves_played = []
      player = self.current_player
      opponent = self.current_player % 2 + 1

      is_endgame, p0_score, p1_score = check_endgame(current_state, player, opponent)
      consecutive_passes = 0  # Counter for consecutive passes

      while not is_endgame:
          move = self.rollout_policy(current_state, player)
          moves_played.append((player, move))

          if move is not None:
              execute_move(current_state, move, player)
              consecutive_passes = 0  # Reset pass counter
          else:
              consecutive_passes += 1

          # Check for endgame condition: two consecutive passes
          if consecutive_passes >= 2:
              break  # End the game

          # Swap players
          player, opponent = opponent, player
          is_endgame, p0_score, p1_score = check_endgame(current_state, player, opponent)

      # Determine the winner
      winner = 1 if p0_score > p1_score else 2
      return winner, moves_played


    def backpropagate(self, winner, moves_played):
      node = self
      while node is not None:
          node.visits += 1
          # The player who made the move leading to this node is the opponent of node.current_player
          player_who_moved = 1 if node.current_player == 2 else 2
          if player_who_moved == winner:
              node.wins += 1
          # Update RAVE statistics
          for player, move in moves_played:
              if move is not None:
                  node.rave_visits[move] = node.rave_visits.get(move, 0) + 1
                  if player == player_who_moved and player == winner:
                      node.rave_wins[move] = node.rave_wins.get(move, 0) + 1
          node = node.parent
