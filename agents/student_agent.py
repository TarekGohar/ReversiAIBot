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

STEP_TIME_LIMIT = 1.99

# Get the corners of the game board
def get_board_corners(board):
    return [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]

# Get the edges of the game board 
def get_board_edges(board):
    return [(0, i) for i in range(1, board.shape[1] - 1)] + \
                [(board.shape[0] - 1, i) for i in range(1, board.shape[1] - 1)] + \
                [(i, 0) for i in range(1, board.shape[0] - 1)] + \
                [(i, board.shape[1] - 1) for i in range(1, board.shape[0] - 1)]

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    print(f"Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")

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

def greedy_flips_score(board, move, player):
    """
    Compute a score based on the number of discs flipped by the move.
    """
    return count_capture(board, move, player)  # Assuming `count_capture` exists

def is_adjacent_to_corner(move, board):
    """
    Check if a move is adjacent to a corner.
    """
    corners = get_board_corners(board)
    for corner in corners:
        if abs(corner[0] - move[0]) <= 1 and abs(corner[1] - move[1]) <= 1:
            return True
    return False

def evaluate_board(board_state):
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

    # Endgame evaluation
    is_endgame, p0_score, p1_score = check_endgame(board_state, 1, 2)
    if is_endgame:
        return np.inf if p0_score > p1_score else -np.inf

    # Define important spots
    corners = get_board_corners(board_state)
    edges = get_board_edges(board_state)
    edges = list(set(edges) - set(corners))  # Remove corners from edges

    # Initialize scores
    max_score = 0
    min_score = 0

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
                score += 200
            else:
                score += 10


            # Exposed discs
            if is_stable(board_state, row, col):
                score += 7

            if is_at_risk(board_state, row, col):
                score -= 5



            # Add score to the appropriate player
            if current_player == 1:
                max_score += score
            else:
                min_score += score

    return max_score - min_score



class Node:
    def __init__(self, key, value, size):
        self.key = key
        self.value = value
        self.size = size
        self.prev = None
        self.next = None


class MemoryLimitedLRUCache:
    def __init__(self, memory_limit_mb):
        """
        Initialize the cache with a memory limit specified in megabytes (MB).
        
        Args:
            memory_limit_mb (int): Maximum memory in megabytes for the cache.
        """
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert MB to bytes
        self.current_memory = 0
        self.cache = {}
        self.head = Node(0, 0, 0)  # Dummy head
        self.tail = Node(0, 0, 0)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        """
        Remove a node from the doubly linked list.
        """
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_head(self, node):
        """
        Add a node right after the dummy head.
        """
        next_node = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = next_node
        next_node.prev = node

    def _get_size(self, key, value):
        """
        Estimate the memory usage of a key-value pair.
        
        Args:
            key: The key object.
            value: The value object.
        
        Returns:
            int: The estimated size in bytes.
        """
        key_size = sys.getsizeof(key)
        if isinstance(key, np.ndarray):
            key_size += key.nbytes
        value_size = sys.getsizeof(value)
        if isinstance(value, np.ndarray):
            value_size += value.nbytes
        return key_size + value_size

    def _evict(self):
        """
        Evict the least recently used item to free up memory.
        """
        while self.current_memory > self.memory_limit:
            lru_node = self.tail.prev
            if lru_node == self.head:
                break  # No items left to evict
            print(f"Evicting key: {lru_node.key}, size: {lru_node.size} bytes")
            self._remove(lru_node)
            del self.cache[lru_node.key]
            self.current_memory -= lru_node.size

    def get(self, key):
        """
        Retrieve a value from the cache.
        
        Args:
            key: The key to retrieve.
        
        Returns:
            The value associated with the key, or -1 if not found.
        """
        hashed_key = key.tobytes() if isinstance(key, np.ndarray) else key
        if hashed_key in self.cache:
            node = self.cache[hashed_key]
            # Move the accessed node to the head (most recently used)
            self._remove(node)
            self._add_to_head(node)
            return node.value
        return None

    def put(self, key, value):
        """
        Add a key-value pair to the cache.
        
        Args:
            key: The key of the item.
            value: The value of the item.
        """
        hashed_key = key.tobytes() if isinstance(key, np.ndarray) else key
        size = self._get_size(key, value)

        if hashed_key in self.cache:
            node = self.cache[hashed_key]
            # Update the value and move to the head
            self.current_memory -= node.size
            node.value = value
            node.size = size
            self.current_memory += size
            self._remove(node)
            self._add_to_head(node)
        else:
            # Add a new node
            new_node = Node(hashed_key, value, size)
            self.cache[hashed_key] = new_node
            self._add_to_head(new_node)
            self.current_memory += size

        # Evict items if over memory limit
        self._evict()

    def display(self):
        """
        Display the current state of the cache.
        """
        node = self.head.next
        items = []
        while node != self.tail:
            items.append((node.key, node.value, node.size))
            node = node.next
        print(f"Cache (Total Memory: {self.current_memory / (1024 * 1024):.2f}/{self.memory_limit / (1024 * 1024):.2f} MB):", items)





@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    self.cache = MemoryLimitedLRUCache(memory_limit_mb=450)
    self.largest_moves = 0


  def step(self, chess_board, player, opponent):
    # Start clock
    start_time = time.time()

    depth = 2
    best_move = None

    #  Iterative Deepening
    while time.time() - start_time < STEP_TIME_LIMIT and depth <= 100:
            try:
                _, move = minimax_alpha_beta(
                    chess_board, depth, -np.inf, np.inf, player, start_time, self.cache
                )
                if move:
                    best_move = move  # Update best move if valid
                print(best_move)
                depth += 1
            except Exception as e:
                print(f"Error at depth {depth}: {e}")
                break

        

    # Check memory usage
    monitor_memory()
    
    print(depth)
    if best_move == None:
        input()
    return best_move

def minimax_alpha_beta(board, depth, alpha, beta, player, start_time, cache):
    valid_moves = get_valid_moves(board, player)
    corners = get_board_corners(board)
    edges = get_board_edges(board)

    valid_moves.sort(key=lambda move: (
                    0 if move in corners else
                    1 if move in edges else
                    2 -
                    0.5 * greedy_flips_score(board, move, player) -
                    0.2 * 1 if is_adjacent_to_corner(move, board) else 0
                ))


    # Base case
    if depth == 0 or not valid_moves or time.time() - start_time >= STEP_TIME_LIMIT:
        cached_score = cache.get(board)

        if cached_score:
            # print("Used cache with value ", cached_score)
            return cached_score, None
        

        score = evaluate_board(board)
        cache.put(board.tobytes(), score)

        return score, None

    best_move = None

    if player == 1:  # Maximizing player
        max_eval = -np.inf
        for move in valid_moves:
            inner_board = deepcopy(board)
            execute_move(inner_board, move, player)

            # Recursive call
            inner_board_eval, _ = minimax_alpha_beta(
                inner_board, depth - 1, alpha, beta, 3 - player, start_time, cache
            )

            if inner_board_eval > max_eval:
                max_eval = inner_board_eval
                best_move = move
            alpha = max(alpha, inner_board_eval)
            if beta <= alpha:
                # print(f"Pruning at move {move}, depth={depth}")
                break
        return max_eval, best_move

    else:  # Minimizing player
        min_eval = np.inf
        for move in valid_moves:
            inner_board = deepcopy(board)
            execute_move(inner_board, move, player)

            # Recursive call
            inner_board_eval, _ = minimax_alpha_beta(
                inner_board, depth - 1, alpha, beta, 3 - player, start_time, cache
            )

            if inner_board_eval < min_eval:
                min_eval = inner_board_eval
                best_move = move
            beta = min(beta, inner_board_eval)
            if beta <= alpha:
                # print(f"Pruning at move {move}, depth={depth}")
                break
        return min_eval, best_move
