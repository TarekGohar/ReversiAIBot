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
    memory_mb = memory_info.rss / (1024 * 1024)
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

# Get zeroes in board
def count_zeros(board):
    return np.count_nonzero(board == 0)


def exposes_corner(board, move, corners, player):
    """
    Check if making this move exposes a corner to the opponent.
    """
    inner_board = deepcopy(board)
    execute_move(inner_board, move, player)
    opponent_moves = get_valid_moves(inner_board, 3 - player)
    for opp_move in opponent_moves:
        if opp_move in corners:
            return True
    return False

def beta(visits, k):
    """
    Calculate the beta value for a given state.
    Args:
        visits (int): Total number of visits to the state (N(s)).
        k (float): Equivalence parameter for RAVE. Default is 1e-5.
    Returns:
        float: Beta value, balancing RAVE and MC values.
    """
    return math.sqrt(k / (3 * visits + k)) if visits > 0 else 1

def dynamic_rollout_depth(board_state):
    empty_count = np.count_nonzero(board_state == 0)
    total_spots = empty_count + np.count_nonzero(board_state == 1) + np.count_nonzero(board_state == 2)

    if empty_count/total_spots > .5:
        return 12  # Early game
    elif empty_count/total_spots > .2:
        return 12  # Midgame
    else:
        return 12  # Late game

# Helper functions
def is_stable(board, row, col):
    """
    Check if the piece at (row, col) is stable in a Reversi game.
    
    Args:
        board (np.array): 2D array representing the Reversi board. 
                          0 for empty, 1 for black, -1 for white.
        row (int): Row index of the piece to check.
        col (int): Column index of the piece to check.

    Returns:
        bool: True if the piece is stable, False otherwise.
    """
    # Directions to check (horizontal, vertical, diagonal)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    piece = board[row, col]
    
    # An empty cell or invalid index cannot be stable
    if piece == 0:
        return False
    
    rows, cols = board.shape
    
    def is_edge_or_corner(r, c):
        """Check if a position is on the edge or corner."""
        return r in {0, rows - 1} or c in {0, cols - 1}
    
    # Pieces on corners are always stable
    if is_edge_or_corner(row, col):
        return True
    
    def direction_stable(r, c, dr, dc):
        """
        Check stability in one direction (dr, dc).
        """
        stable = True
        while 0 <= r < rows and 0 <= c < cols:
            if board[r, c] != piece and board[r, c] != 0:
                stable = False
                break
            r += dr
            c += dc
        return stable
    
    # Check stability in all directions
    for dr, dc in directions:
        if not direction_stable(row + dr, col + dc, dr, dc):
            return False

    return True

def is_at_risk(board, row, col):
    """
    Check if the piece at (row, col) is at risk of being flipped in a Reversi game.
    
    Args:
        board (np.array): 2D array representing the Reversi board.
                          0 for empty, 1 for black, -1 for white.
        row (int): Row index of the piece to check.
        col (int): Column index of the piece to check.

    Returns:
        bool: True if the piece is at risk, False otherwise.
    """
    # Directions to check (horizontal, vertical, diagonal)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    piece = board[row, col]
    
    # An empty cell or invalid index cannot be at risk
    if piece == 0:
        return False
    
    opponent = -piece
    rows, cols = board.shape
    
    def can_flip_in_direction(r, c, dr, dc):
        """
        Check if the piece can be flipped in one direction (dr, dc).
        """
        r += dr
        c += dc
        has_opponent_pieces = False
        
        while 0 <= r < rows and 0 <= c < cols:
            if board[r, c] == opponent:
                has_opponent_pieces = True  # Found opponent's piece
            elif board[r, c] == piece:
                return has_opponent_pieces  # Stable sequence ends with player's piece
            else:
                break  # Reached an empty square or the edge
            r += dr
            c += dc
        
        return False  # No valid flip found in this direction
    
    # Check all directions
    for dr, dc in directions:
        if can_flip_in_direction(row, col, dr, dc):
            return True

    return False

def evaluate_board_state(board_state, player):
    """
    Evaluate the board state relative to an initial board state for better decision-making.
    
    Args:
        board_state (np.ndarray): 2D array representing the board state.
        player (int): The player for whom the evaluation is being done (1 or 2).
        initial_board_state (np.ndarray, optional): The initial board state for comparison.
        
    Returns:
        tuple: (player_score, opponent_score, is_endgame)
    """
    rows, cols = board_state.shape

    # Define important spots
    corners = get_board_corners(board_state)
    edges = get_board_edges(board_state)
    edges = list(set(edges) - set(corners))  # Remove corners from edges

    # Initialize scores
    player_score = 0
    opponent_score = 0

    # Endgame evaluation
    # if empty_spots < 10:
    _, p0_score, p1_score = check_endgame(board_state, player, 3 - player)
    # if is_endgame:
    #     return (1, 0, p0_score, p1_score, True) if p0_score > p1_score else (0, 1, p0_score, p1_score, True)

    # Edge and corner considerations
    for row in range(rows):
        for col in range(cols):
            if board_state[row, col] == 0:
                continue

            current_player = board_state[row, col]
            score = 0

            # Occupancy of important spots
            if (row, col) in corners:
                score += 4000
            elif (row, col) in edges:
                score += 250
            else:
                score += 10


            # Exposed discs
            if is_stable(board_state, row, col):
                score += 20

            if is_at_risk(board_state, row, col):
                score -= 25



            # Add score to the appropriate player
            if current_player == player:
                player_score += score
            else:
                opponent_score += score

    return player_score, opponent_score, p0_score, p1_score, False

@register_agent("rave_agent")
class RaveAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(RaveAgent, self).__init__()
    self.name = "RaveAgent"

  def step(self, chess_board, player, opponent):
    # Start clock
    start_time = time.time()

    # Create the root node
    root_node = MCTSNode(chess_board, player)

    # Run MCTS/RAVE search
    move = root_node.mcts_search(player, start_time)

    # Check memory usage
    monitor_memory()

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

        self.best_child_cache = None
        self.best_score_cache = None

    def mcts_search(self, player, start_time, time_limit=1.99):
        while (time.time() - start_time) < time_limit:
            node = self
            while node.children:
                node = node.best_child()

            node.expand()
            if node.children:
                node = np.random.choice(node.children)

            winner, moves_played = node.rollout()
            node.backpropagate(winner, moves_played)

        return max(self.children, key=lambda c: c.visits).move if self.children else None
    
    # Expand the node by adding child nodes for each valid move
    def expand(self):
        if not self.children:
            next_player = 3 - self.current_player
            moves = get_valid_moves(self.board_state, self.current_player)

            for move in moves:
                new_board = deepcopy(self.board_state)
                execute_move(new_board, move, self.current_player)
                child_node = MCTSNode(new_board, next_player, parent=self, move=move)
                self.children.append(child_node)

        # Invalidate cached best child
        self.best_child_cache = None
        self.best_score_cache = None



    def rollout_policy(self, board_state, player):
        valid_moves = get_valid_moves(board_state, player)

        # If no valid moves, pass
        if not valid_moves:
            return None

        # Define board corners and edges
        corners = get_board_corners(board_state)

        for corner in corners:
            if corner in valid_moves:
                return corner
        
        most_captures = 0
        best_move = None

        for move in valid_moves:
            if exposes_corner(board_state, move, corners, player):
                continue
            
            move_value = count_capture(board_state, move, player)
            if move_value > most_captures:
                most_captures = move_value
                best_move = move

        return move

    def rollout(self):
        """
        Perform a rollout simulation from the current state, dynamically adjusting the depth
        based on the number of empty spaces on the board.
        Returns:
            winner (int): The winner determined by the simulation.
            moves_played (list): List of moves played during the rollout.
            steps (int): Number of steps taken to reach the outcome.
        """
        current_state = deepcopy(self.board_state)
        moves_played = []
        player = self.current_player
        opponent = 3 - self.current_player

        # Determine rollout depth dynamically based on empty tiles
        rollout_depth = dynamic_rollout_depth(current_state)  # Scale with empty tiles

        steps = 0  # Track number of steps in the rollout

        # Simulate game
        for _ in range(rollout_depth):
            move = self.rollout_policy(current_state, player)
            if move is None:  # No valid moves, game may be over
                break
            moves_played.append((player, move))
            execute_move(current_state, move, player)

            # Check if the game has ended
            is_endgame, _, _ = check_endgame(current_state, player, opponent)
            if is_endgame:
                break
            # Swap turns
            player, opponent = opponent, player

        # Evaluate board states for comparison
        p0_score_start, p1_score_start, p0_tiles_start, p1_tiles_start, _ = evaluate_board_state(self.board_state, self.current_player)
        p0_score, p1_score, p0_tiles, p1_tiles, is_endgame = evaluate_board_state(current_state, player)

        # Determine winner based on current state
        winner = 0
        if (p0_score - p0_score_start) > (p1_score - p1_score_start) and p0_tiles > p1_tiles:
            winner = self.current_player
        elif (p1_score - p1_score_start) > (p0_score - p0_score_start) and p1_tiles > p0_tiles:
            winner = 3 - self.current_player

        return winner, moves_played


    # Backpropagate the results of the simulation
    def backpropagate(self, winner, moves_played):
        """
        Backpropagate the results of the simulation up the tree, incorporating the number of steps.
        Args:
            winner (int): The player who won the simulation (1 or 2).
            moves_played (list): List of (player, move) pairs played during the simulation.
            steps (int): Number of steps taken to reach the outcome.
        """
        node = self
        while node is not None:
            # Increment visits
            node.visits += 1
            if node.parent and winner == 3 - node.current_player:
                node.wins += 1

            # Update RAVE statistics
            for player, move in moves_played:
                if move is not None:
                    node.rave_visits[move] = node.rave_visits.get(move, 0) + 1
                    if player == winner:
                        node.rave_wins[move] = node.rave_wins.get(move, 0) + 1

            # Move up the tree
            node.best_child_cache = None
            node.best_score_cache = None

            node = node.parent


    # Check if the node has children
    def has_children(self):
        return len(self.children) > 0

    # Select the best child node based on the UCT/RAVE formula
    def best_child(self, exploration=1.41, k=1000):
        """
        Select the best child node using UCT and RAVE statistics.
        """
        if self.best_child_cache is not None:
            return self.best_child_cache

        best_score = -np.inf
        best_child = None

        for child in self.children:
            if child.visits == 0:
                return child

            beta_value = beta(self.visits, k)
            q_value = child.wins / child.visits
            rave_wins = child.rave_wins.get(child.move, 0)
            rave_visits = child.rave_visits.get(child.move, 0)
            rave_q = rave_wins / rave_visits if rave_visits > 0 else 0

            combined_q = (1 - beta_value) * q_value + beta_value * rave_q
            uct_value = combined_q + exploration * math.sqrt(math.log(self.visits) / child.visits)

            if uct_value > best_score:
                best_score = uct_value
                best_child = child

        self.best_child_cache = best_child
        self.best_score_cache = best_score
        return best_child

        return best_child
