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

STEP_TIME_LIMIT = 1.98

# Helper functions

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def get_board_corners(board):
    return [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]

def get_board_edges(board):
    return [(0, i) for i in range(1, board.shape[1] - 1)] + \
                [(board.shape[0] - 1, i) for i in range(1, board.shape[1] - 1)] + \
                [(i, 0) for i in range(1, board.shape[0] - 1)] + \
                [(i, board.shape[1] - 1) for i in range(1, board.shape[0] - 1)]

def greedy_flips_score(board, move, player):
    """
    Compute a score based on the number of discs flipped by the move.
    """
    return count_capture(board, move, player)  # Assuming `count_capture` exists

def is_adjacent_to_corner(board, move):
    """
    Check if a move is adjacent to a corner.
    """
    corners = get_board_corners(board)
    for corner in corners:
        if abs(corner[0] - move[0]) <= 1 and abs(corner[1] - move[1]) <= 1:
            return True
    return False

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


# Main functions for MiniMax

def evaluate_board(board_state):
    """
    Evaluation function that prioritizes positional heuristics in the early and midgame,
    and maximizes piece count in the final quarter of the game.

    Args:
        board_state (np.ndarray): 2D array representing the board state.
    
    Returns:
        float: Evaluation score for the current board state.
    """
    rows, cols = board_state.shape

    # Endgame evaluation
    is_endgame, p0_score, p1_score = check_endgame(board_state, 1, 2)
    if is_endgame:
        return np.inf if p0_score > p1_score else -np.inf

    # Count total and empty spaces
    total_spaces = rows * cols
    empty_spaces = np.sum(board_state == 0)

    # Define important spots
    corners = get_board_corners(board_state)
    edges = get_board_edges(board_state)
    edges = list(set(edges) - set(corners))  # Remove corners from edges

    # Piece counts
    player_pieces = np.sum(board_state == 1)
    opponent_pieces = np.sum(board_state == 2)

    # Final quarter: Maximize piece count
    if empty_spaces <= total_spaces / 4:
        return player_pieces - opponent_pieces  # Maximize piece count directly

    # Early and midgame: Use positional heuristics
    max_score = 0
    min_score = 0

    

    # Evaluate positional factors
    for row in range(rows):
        for col in range(cols):
            if board_state[row, col] == 0:
                continue

            current_player = board_state[row, col]
            score = 0

            # Occupancy of important spots
            if (row, col) in corners:
                score += 500
            elif (row, col) in edges:
                score += 40
            else:
                score += 10

            # Stability and risk
            if is_stable(board_state, row, col):
                score += 25
            if is_at_risk(board_state, row, col):
                score -= 20

            # Add score to the appropriate player
            if current_player == 1:
                max_score += score
            else:
                min_score += score

    # Combine positional scores
    return max_score - min_score

def minimax_alpha_beta(board, depth, alpha, beta, player, start_time, cache):
    valid_moves = get_valid_moves(board, player)
    corners = get_board_corners(board)
    edges = get_board_edges(board)

    valid_moves.sort(key=lambda move: (
        0 if move in corners else
        1 if move in edges else
        2 if exposes_corner(board, move, corners, player) else
        3 - 0.5 * greedy_flips_score(board, move, player)
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

def determine_algorithm(board_state):
    board_size = len(board_state)
    empty_spots = np.count_nonzero(board_state == 0)
    total_spots = board_size ** 2

    game_left = empty_spots/total_spots

    if game_left > .85 and (board_size == 10 or board_size == 12):
        return "IDMM"
    else:
        return "MCTS"

@register_agent("hybrid_agent")
class HybridAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(HybridAgent, self).__init__()
    self.name = "HybridAgent"
    self.cache = MemoryLimitedLRUCache(400)

  def step(self, chess_board, player, opponent):
    start_time = time.time()

    # Call periodically in your program
    # monitor_memory()

    # algorithm to run
    algorithm = determine_algorithm(chess_board)

    depth = 0
    best_move = None

    valid_moves = get_valid_moves(chess_board, player)
    corners = get_board_corners(chess_board)

    # return corners immediately if possible
    for corner in corners:
        if corner in valid_moves:
            return corner

    while time.time() - start_time < STEP_TIME_LIMIT:
        if algorithm == "MCTS":
            node = MCTSNode(chess_board, player)
            best_move = node.mcts_search(player, start_time, STEP_TIME_LIMIT)
        else:
            _, move = minimax_alpha_beta(
                        chess_board, depth, -np.inf, np.inf, player, start_time, self.cache
                    )
            if move:
                best_move = move  # Update best move if valid
            depth += 1


    return best_move
    
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

    def evaluate_board(self, board, player, player_score, opponent_score):
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
        # Value the corner moves
        corners = get_board_corners(board)
        corner_score = sum(1 for corner in corners if board[corner] == player) * 10
        corner_penalty = sum(1 for corner in corners if board[corner] == 3 - player) * 10

        # Valuye the number of moves the opponent can make
        opponent_moves = len(get_valid_moves(board, 3 - player))

        # Return combined scores
        return player_score - opponent_score + corner_score - corner_penalty - opponent_moves

    def mcts_search(self, player, start_time, time_limit=1.9):

        corners = get_board_corners(self.board_state)
        edges = get_board_edges(self.board_state)
      
        while time.time() - start_time < time_limit:
            # set node equal to root
            node = self

            # check for endgame
            is_endgame, _, _ = check_endgame(node.board_state, player, 3 - player)

            # Selection
            while node.has_children() and not is_endgame:
                # keep finding best candidate until leaf node reached
                node = node.best_child()

            # Expansion
            if not is_endgame:
                node.expand(corners, edges)

                if len(node.children) > 0:
                    # pick first child since ordered by importance
                    node = node.children[0]

            # Simulation
            winner, moves_played = node.rollout()

            # Backpropagation
            node.backpropagate(winner, moves_played)

        # Return the move with the most visits as per MCTS
        if self.children:
            best_move = max(self.children, key=lambda c: c.visits).move
        else:
            # If no children were expanded, return a random valid move
            best_move = random_move(self.board_state, player)
        return best_move

    def expand(self, corners, edges):
        # verify no children
        if not self.children:
            valid_moves = get_valid_moves(self.board_state, self.current_player)

            if not valid_moves:
                # No valid moves, so the player must pass
                valid_moves = [None]  # Represent pass move with None

            # order by importance
            valid_moves.sort(key=lambda move: (
                0 if move in corners else
                1 if move in edges else
                2
            ))

            for move in valid_moves:
                # create new board state per move
                new_board = deepcopy(self.board_state)

                if move is not None:
                    execute_move(new_board, move, self.current_player)

                # Switch to the next player
                next_player = 3 - self.current_player

                # create and add node
                child_node = MCTSNode(new_board, next_player, parent=self, move=move)
                self.children.append(child_node)

    def has_children(self):
        return len(self.children) > 0

    def best_child(self, exploration=1.41):
        best_score = -np.inf
        best_child = None

        for child in self.children:
            # return unvisited nodes first
            if child.visits == 0:
                return child
            
            # RAVE stats
            rave_visit = child.rave_visits.get(child.move, 0)
            rave_win = child.rave_wins.get(child.move, 0)
            rave_q = rave_win / rave_visit if rave_visit > 0 else 0

            # beta for balance between exploitation and exploration
            beta = rave_visit / (rave_visit + child.visits + 1e-4)

            q_value = child.wins / child.visits
            
            # MC-RAVE + UCT equation
            uct_value = (1 - beta) * q_value + beta * rave_q + exploration * math.sqrt(math.log(self.visits) / child.visits)

            if uct_value > best_score:
                best_score = uct_value
                best_child = child

        return best_child

    def rollout_policy(self, board_state, player):
        valid_moves = get_valid_moves(board_state, player)

        if not valid_moves:
            return None  # No valid moves available then pass turn

        best_move = None
        best_score = float('-inf')

        # Prioritize corners, maximize flips and minimize opponent potential moves
        for move in valid_moves:
            simulated_board = deepcopy(board_state)
            execute_move(simulated_board, move, player)

            # get scores
            player_score = np.count_nonzero(simulated_board == player)
            opponent_score = np.count_nonzero(simulated_board == 3 - player)

            # get overall state score
            move_score = self.evaluate_board(simulated_board, player, player_score, opponent_score)

            # keep best scored move
            if move_score > best_score:
                best_score = move_score
                best_move = move

        # Return the best move found
        return best_move if best_move else np.random.choice(valid_moves)

    def rollout(self):
        # create copy of state to simulate on
        current_state = deepcopy(self.board_state)

        # list of moves made during simulation
        moves_played = []

        # current player
        player = self.current_player

        # current opponent
        opponent = 3 - self.current_player

        # check for terminal state
        is_endgame, p0_score, p1_score = check_endgame(current_state, self.current_player, 3 - self.current_player)
        stalemate_counter = 0  # flag for stalemate

        while not is_endgame:
            move = self.rollout_policy(current_state, player)
            moves_played.append((player, move))

            if move is not None:
                execute_move(current_state, move, player)
                stalemate_counter = 0  # Reset counter
            else:
                stalemate_counter += 1

            # Check for stalemate when both sides cannot move
            if stalemate_counter >= 2:
                break  # End the game

            # Swap players
            player, opponent = opponent, player
            
            # check for endgame
            is_endgame, p0_score, p1_score = check_endgame(current_state, self.current_player, 3 - self.current_player)

        # Determine the winner, ties and losses count as losses
        winner = self.current_player if p0_score > p1_score else 3 - self.current_player
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

# MinMax Node Class
class MinMaxNode:
    def __init__(self, key, value, size):
        self.key = key
        self.value = value
        self.size = size
        self.prev = None
        self.next = None

# Class to handle IDS caching
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
        self.head = MinMaxNode(0, 0, 0)  # Dummy head
        self.tail = MinMaxNode(0, 0, 0)  # Dummy tail
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
            # print(f"Evicting key: {lru_node.key}, size: {lru_node.size} bytes")
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
            new_node = MinMaxNode(hashed_key, value, size)
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
        # print(f"Cache (Total Memory: {self.current_memory / (1024 * 1024):.2f}/{self.memory_limit / (1024 * 1024):.2f} MB):", items)

