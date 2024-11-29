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
    # print(f"Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")
        
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

def beta(visits, k=200):
    """
    Calculate the beta value for a given state.
    Args:
        visits (int): Total number of visits to the state (N(s)).
        k (float): Equivalence parameter for RAVE. Default is 1e-5.
    Returns:
        float: Beta value, balancing RAVE and MC values.
    """
    return math.sqrt(k / (3 * visits + k)) if visits > 0 else 1


# Helper functions

def count_edge_occupancy(board_state, player):
    """
    Counts the number of edges occupied by the given player and their opponent.
    
    Args:
        board_state (np.ndarray): 2D grid representing the game board.
        player (int): The player for whom the edge occupancy is being calculated (1 or 2).
        
    Returns:
        tuple: (player_edges, opponent_edges)
            player_edges (int): Number of edges occupied by the player.
            opponent_edges (int): Number of edges occupied by the opponent.
    """
    opponent = 3 - player  # Determine the opponent player
    player_edges = 0
    opponent_edges = 0

    rows, cols = board_state.shape

    # Define edge cells: top row, bottom row, left column, right column
    top_row = board_state[0, :]
    bottom_row = board_state[rows - 1, :]
    left_column = board_state[:, 0]
    right_column = board_state[:, cols - 1]

    # Count player and opponent pieces on the edges
    for edge in [top_row, bottom_row, left_column, right_column]:
        player_edges += np.sum(edge == player)
        opponent_edges += np.sum(edge == opponent)

    return player_edges, opponent_edges

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

    # Count remaining tiles
    empty_spots = count_zeros(board_state)

    # Endgame evaluation
    # if empty_spots < 10:
    is_endgame, p0_score, p1_score = check_endgame(board_state, player, 3 - player)
    if is_endgame:
        return (1, 0, True) if p0_score > p1_score else (0, 1, True)

    # Edge and corner considerations
    for row in range(rows):
        for col in range(cols):
            if board_state[row, col] == 0:
                continue

            current_player = board_state[row, col]
            score = 0

            # Occupancy of important spots
            if (row, col) in corners:
                score += 10000
            elif (row, col) in edges:
                score += 125
            else:
                score += 10


            # Exposed discs
            if is_stable(board_state, row, col):
                score += 5

            if is_at_risk(board_state, row, col):
                score -= 12



            # Add score to the appropriate player
            if current_player == player:
                player_score += score
            else:
                opponent_score += score

    return player_score, opponent_score, False

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
        """
        Run the MCTS/RAVE search algorithm to select the best move.
        This function performs a Monte Carlo Tree Search (MCTS) with Rapid Action Value Estimation (RAVE) to determine the optimal move for the given player from the root state. It prioritizes corner moves if available and iteratively expands the search tree within the given time limit.
        Args:
            root_state (np.ndarray): The current state of the game board.
            player (int): The player for whom the move is being selected.
            start_time (float): The start time of the search process.
            time_limit (float, optional): The maximum time allowed for the search in seconds. Default is 1.9 seconds.
        Returns:
            tuple: The best move determined by the MCTS/RAVE algorithm, or a random valid move if no children were expanded, or None if no valid moves are available.
        """

        corners = get_board_corners(self.board_state)
        edges = get_board_edges(self.board_state)
        moves = get_valid_moves(self.board_state, player)

        # If corner move available, play it
        # for move in moves:
        #     if move in corners:
        #         return move

        depth_sum = 0
        iterations = 0
        while (time.time() - start_time) < time_limit:
            node = self
            is_endgame, _, _ = check_endgame(node.board_state, player, 3 - player)
            # print("iteration", iterations)

            # Selection: Select the best child node until a leaf node is reached
            depth = 0
            while node.has_children() and not is_endgame:
                node = node.best_child()
                depth += 1
            
            depth_sum += depth
            iterations += 1

            # Expansion: Expand the tree by adding a new child node if the current node is not a terminal state
            if not is_endgame:
                node.expand(corners, edges)
                if node.children:
                    # Select a random child node to explore
                    node = node.children[0]

            # Simulation: Simulate a game from the current node to determine the winner
            winner, moves_played = node.rollout()

            # Backpropagation
            node.backpropagate(winner, moves_played)

        print(depth_sum / iterations)
        print(depth_sum)
        print(iterations)
        # Return the move with the most promising child
        if self.children:
            best_move = None
            max_node = -np.inf
            for child in self.children:
                print(child.move, child.wins, child.visits)
                if child.visits > max_node:
                    max_node = child.visits
                    best_move = child.move


            best_move =  max(self.children, key=lambda c: c.visits).move
        else:
            # If no children were expanded, return a random valid move
            if moves:
                best_move = moves[np.random.randint(len(moves))]
            else:
                best_move = None  # No valid moves, must pass

        # input()
        return best_move
    
    # Expand the node by adding child nodes for each valid move
    def expand(self, corners, edges):
        if not self.children:
            next_player = 3 - self.current_player
            moves = get_valid_moves(self.board_state, self.current_player)

            # Prioritize moves: corners > edges > others
            moves.sort(key=lambda move: (
                0 if move in corners else  # Highest priority for corners
                1 if move in edges else    # Second priority for edges
                2                          # Lowest priority for others
            ))

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
        # corners = get_board_corners(board_state)

        # for corner in corners:
        #     if corner in valid_moves:
        #         return corner

        return valid_moves[np.random.randint(len(valid_moves))]

    def rollout(self):
        """
        Perform a rollout simulation from the current state, dynamically adjusting the depth
        based on the number of empty spaces on the board.
        Args:
            max_rollout_depth (int): The initial maximum depth of the rollout.
        Returns:
            winner (int): The winner determined by the simulation.
            moves_played (list): List of moves played during the rollout.
        """
        current_state = deepcopy(self.board_state)
        moves_played = []
        player = self.current_player
        opponent = 3 - self.current_player
        rollout_depth = 6  # Minimum depth at the start of the game

        for _ in range(rollout_depth):
            move = self.rollout_policy(current_state, player)
            if move is None:
                break  # No valid moves, game may be over
            moves_played.append((player, move))
            execute_move(current_state, move, player)
            is_endgame, _, _ = check_endgame(current_state, player, opponent)
            if is_endgame:
                break
            player, opponent = opponent, player

        # Approximate winner based on current board state
        p0_score_start, p1_score_start, _ = evaluate_board_state(self.board_state, player)
        p0_score, p1_score, is_endgame = evaluate_board_state(current_state, player)
        
        winner = 0

        if is_endgame:
            if p0_score > p1_score:
                winner = player
            elif p1_score > p0_score:
                winner = 3 - player
            else:
                winner = 0
        else:
            p0_change = p0_score - p0_score_start
            p1_change = p1_score - p1_score_start
            # if p0_score > p1_score and (p0_change) >= rollout_depth * 10 and (p1_score - p1_score_start) <= p0_change:
            if p0_score > p1_score and (p0_change) >= rollout_depth * 5 and abs(p1_change) <= rollout_depth * 30:
                winner = player
            elif p1_score > p0_score and (p1_change) >= rollout_depth * 5 and abs(p0_change) <= rollout_depth * 30:
                winner = 3 - player
            else:
                winner = 0

        return winner, moves_played

    # Backpropagate the results of the simulation
    def backpropagate(self, winner, moves_played):
        """
        Backpropagate the results of the simulation up the tree.
        Args:
            winner (int): The player who won the simulation (1 or 2).
            moves_played (list): List of (player, move) pairs played during the simulation.
        """
        node = self
        while node is not None:
            # Increment visits
            node.visits += 1
            
            # Determine if this node's action resulted in a win
            if node.parent is not None:  # Skip the root node
                player_who_moved = 3 - node.current_player  # Parent's perspective
                if player_who_moved == winner:
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
    def best_child(self, exploration=np.sqrt(2), k=500):
        best_score = -np.inf
        best_child = None

        for child in self.children:
            if child.visits == 0:
                return child  # Prioritize unexplored nodes

            # Calculate beta using the RAVE formula
            beta_value = beta(self.visits, k)

            # Calculate combined Q* value
            q_value = child.wins / child.visits
            rave_wins = child.rave_wins.get(child.move, 0)
            rave_visits = child.rave_visits.get(child.move, 0)
            rave_q = rave_wins / rave_visits if rave_visits > 0 else 0

            # Combine Q(s, a) and Q~(s, a) with beta
            combined_q = (1 - beta_value) * q_value + beta_value * rave_q

            # Add UCT exploration term
            uct_value = combined_q + exploration * math.sqrt(math.log(self.visits) / child.visits)

            if uct_value > best_score:
                best_score = uct_value
                best_child = child

        # Cache the best child and its score
        self.best_child_cache = best_child
        self.best_score_cache = best_score

        return best_child
