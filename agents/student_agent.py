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

# Prints current memory usage of the program
def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")

# Get the corners of the game board
def get_board_corners(board):
    return [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]

# Get the edges of the game board 
def get_board_edges(board):
    return [(0, i) for i in range(1, board.shape[1] - 1)] + \
                [(board.shape[0] - 1, i) for i in range(1, board.shape[1] - 1)] + \
                [(i, 0) for i in range(1, board.shape[0] - 1)] + \
                [(i, board.shape[1] - 1) for i in range(1, board.shape[0] - 1)]


@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

  def step(self, chess_board, player, opponent):
    # Start clock
    start_time = time.time()

    # Create the root node
    root_node = MCTSNode(chess_board, player)

    # Run MCTS/RAVE search
    move = root_node.mcts_search(player, start_time)

    # Check memory usage
    # monitor_memory()

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

    def mcts_search(self, player, start_time, time_limit=1.9, max_depth=4):
        """
        Run the MCTS/RAVE search algorithm to select the best move, limited by a depth threshold.
        Args:
            player (int): The player for whom the move is being selected.
            start_time (float): The start time of the search process.
            time_limit (float): Maximum allowed time for the search.
            max_depth (int): Maximum depth for the search tree.
        Returns:
            tuple: The best move determined by the MCTS/RAVE algorithm, or a random valid move.
        """
        corners = get_board_corners(self.board_state)
        moves = get_valid_moves(self.board_state, player)

        # If corner move available, play it
        for move in moves:
            if move in corners:
                return move

        while (time.time() - start_time) < time_limit:
            node = self
            depth = 0
            is_endgame, _, _ = check_endgame(node.board_state, player, 3 - player)

            # Selection: Traverse the tree until a leaf node or depth limit
            while node.has_children() and not is_endgame and depth < max_depth:
                node = node.best_child()
                depth += 1

            # If depth limit is reached, stop search and backpropagate
            if depth >= max_depth:
                # Perform a rollout and backpropagation from this point
                winner, moves_played = node.rollout()
                node.backpropagate(winner, moves_played)
                continue

            # Expansion: Expand if not at terminal state
            if not is_endgame:
                node.expand()
                if len(node.children) > 0:
                    # Select a random child node to explore
                    node = node.children[np.random.randint(len(node.children))]
                else:
                    # No moves to expand, backpropagate
                    is_endgame, p0_score, p1_score = check_endgame(node.board_state, player, 3 - player)
                    winner = player if p0_score > p1_score else 3 - player
                    node.backpropagate(winner, [])
                    continue

            # Simulation: Simulate a game from the current node
            winner, moves_played = node.rollout()

            # Backpropagation
            print("Back", winner)
            node.backpropagate(winner, moves_played)

        # Return the best move
        if self.children:
            best_move = max(
                self.children,
                key=lambda c: (
                    0.15 * (c.wins / (c.visits + 1)) +  # Weighted MCTS win rate
                    0.85 * (c.rave_wins.get(c.move, 0) / (c.rave_visits.get(c.move, 0) + 1))  # Weighted AMAF win rate
                )
            ).move
        else:
            # If no children were expanded, return a random valid move
            valid_moves = get_valid_moves(self.board_state, player)
            best_move = valid_moves[np.random.randint(len(valid_moves))] if valid_moves else None

        return best_move

    # Expand the node by adding child nodes for each valid move
    def expand(self):
        # Expand the node by adding child nodes for each valid move
        if not self.children:
            # Get valid moves for the current player
            moves = get_valid_moves(self.board_state, self.current_player)

            # Create child nodes for each valid move
            for move in moves:
                # Create a new board state by executing the move
                new_board = deepcopy(self.board_state)
                if move is not None:
                    execute_move(new_board, move, self.current_player)

                # Switch to the next player
                next_player = 3 - self.current_player
                child_node = MCTSNode(new_board, next_player, parent=self, move=move)
                self.children.append(child_node)

    # Check if the node has children
    def has_children(self):
        return len(self.children) > 0

    # Calculate the beta value for RAVE
    def beta(self):
        k = 1e-5  # RAVE constant
        rave_visits = self.rave_visits.get(self.move, 0)
        return rave_visits / (rave_visits + self.visits + k)

    # Select the best child node based on the UCT/RAVE formula
    def best_child(self, exploration=1.52):
        best_score = -np.inf
        best_child = None

        # Select the child node with the highest UCT/RAVE value
        for child in self.children:
            if child.visits == 0:
                return child
            rave_visit = child.rave_visits.get(child.move, 0)
            beta = self.beta()
            q_value = child.wins / child.visits
            rave_win = child.rave_wins.get(child.move, 0)
            rave_q = rave_win / rave_visit if rave_visit > 0 else 0
            uct_value = (1 - beta) * q_value + beta * rave_q + exploration * math.sqrt(math.log(self.visits) / child.visits)
            if uct_value > best_score:
                best_score = uct_value
                best_child = child
        return best_child


    def rollout_policy(self, board_state, player):
        valid_moves = get_valid_moves(board_state, player)

        # If no valid moves, pass
        if not valid_moves:
            return None

        # Define board corners, edges, and critical squares
        corners = get_board_corners(board_state)
        edges = get_board_edges(board_state)
        size = board_state.shape[0]
        adjacent_to_corners = [
            (0, 1), (1, 0), (1, 1),
            (0, size - 2), (1, size - 2), (1, size - 1),
            (size - 2, 0), (size - 2, 1), (size - 1, 1),
            (size - 2, size - 1), (size - 1, size - 2), (size - 2, size - 2)
        ]

        best_moves = []
        best_score = float('-inf')

        for move in valid_moves:
            move_score = 0
            simulated_board = deepcopy(board_state)

            # Corner priority
            if move in corners:
                return move  # Always take a corner if available

            # Penalize moves adjacent to corners
            if move in adjacent_to_corners:
                move_score -= 100

            # Edge priority
            if move in edges:
                move_score += 50

            # Execute the move and simulate opponent's response
            execute_move(simulated_board, move, player)
            opponent_moves = get_valid_moves(simulated_board, 3 - player)

            # Check for opponent corner capture
            for op_move in opponent_moves:
                if op_move in corners:
                    move_score -= 300  # Penalize if it allows opponent to take a corner

            # Penalize moves that give the opponent many options
            move_score -= len(opponent_moves) * 5

            # Update best move(s)
            if move_score > best_score:
                best_moves = [move]
                best_score = move_score
            elif move_score == best_score:
                best_moves.append(move)

        return best_moves[np.random.randint(len(best_moves))] if best_moves else valid_moves[np.random.randint(len(valid_moves))]

    def rollout(self):
      current_state = deepcopy(self.board_state)
      moves_played = []
      player = self.current_player
      opponent = 3 - self.current_player

      is_endgame, p0_score, p1_score = check_endgame(current_state, player, opponent)
      consecutive_passes = 0  # Counter for stalemate detection

      while not is_endgame:
        move = self.rollout_policy(current_state, player)
        moves_played.append((player, move))

        if move is not None:
            execute_move(current_state, move, player)
            consecutive_passes = 0  # Reset stalemate counter
        else:
            consecutive_passes += 1

        # Check for endgame condition: stalemate
        if consecutive_passes >= 2:
            break  # End the game

        # Check for endgame
        is_endgame, p0_score, p1_score = check_endgame(current_state, player, opponent)

        # Swap players
        player, opponent = opponent, player

      # Determine the winner
      winner = player if p0_score > p1_score else opponent
      return winner, moves_played

    # Backpropagate the results of the simulation
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
                  if player_who_moved == winner:
                      node.rave_wins[move] = node.rave_wins.get(move, 0) + 1
          node = node.parent
