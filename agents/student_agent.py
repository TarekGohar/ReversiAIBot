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

# def evaluate_board_state(board_state, player, initial_board_state=None):
#     """
#     Evaluate the board state relative to an initial board state for better decision-making.
    
#     Args:
#         board_state (np.ndarray): 2D array representing the board state.
#         player (int): The player for whom the evaluation is being done (1 or 2).
#         initial_board_state (np.ndarray, optional): The initial board state for comparison.
        
#     Returns:
#         tuple: (player_score, opponent_score)
#     """
#     rows, cols = board_state.shape

#     # Define important spots
#     corners = get_board_corners(board_state)
#     edges = get_board_edges(board_state)
#     edges = list(set(edges) - set(corners))  # Remove corners from edges

#     # Initialize scores
#     player_score = 0
#     opponent_score = 0

#     # Endgame evaluation
#     total_spots = rows * cols
#     empty_spots = count_zeros(board_state)
#     if empty_spots < 10:  # Only check endgame in the final stages
#         is_endgame, p0_score, p1_score = check_endgame(board_state, player, 3 - player)
#         if is_endgame:
#             if p0_score > p1_score:
#                 return 1, 0
#             elif p1_score > p0_score:
#                 return 0, 1
#             else:
#                 return 0, 0

#     # Mobility factor
#     player_valid_moves = get_valid_moves(board_state, player)
#     opp_valid_moves = get_valid_moves(board_state, 3 - player)

#     player_mobility = len(player_valid_moves)
#     opponent_mobility = len(opp_valid_moves)

#     player_score += player_mobility * 4
#     opponent_score += opponent_mobility * 4

#     def is_stable(board, row, col, p):
#         """Determine if a disc is stable using simplified stability heuristics."""
#         if (row, col) in corners:
#             return True  # Corners are always stable
#         if row == 0 or row == board.shape[0] - 1 or col == 0 or col == board.shape[1] - 1:
#             return True  # Edge pieces are generally stable (simplistic)
#         return False  # Simplify by assuming non-edge, non-corner pieces are unstable

#     def is_exposed(board, row, col):
#         """Determine if a disc is exposed by counting empty neighbors."""
#         directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
#         for dr, dc in directions:
#             r, c = row + dr, col + dc
#             if 0 <= r < board.shape[0] and 0 <= c < board.shape[1] and board[r, c] == 0:
#                 return True  # Exposed if any neighbor is empty
#         return False


#     # Evaluate the current board
#     for row in range(rows):
#         for col in range(cols):
#             if board_state[row, col] == 0:
#                 continue  # Skip empty cells

#             current_player = board_state[row, col]
#             score = 0

#             # Occupancy of important spots
#             if (row, col) in corners:
#                 score += 1000
#             elif (row, col) in edges:
#                 score += 125
#             else:
#                 score += 10

#             # Stable discs
#             if (row, col) in corners or is_stable(board_state, row, col, current_player):
#                 score += 5

#             # Exposed discs
#             if is_exposed(board_state, row, col):
#                 score -= 5

#             # Add score to the appropriate player
#             if current_player == player:
#                 player_score += score
#             else:
#                 opponent_score += score

#     # If initial_board_state is provided, calculate relative improvement
#     if initial_board_state is not None:
#         initial_player_score, initial_opponent_score = evaluate_board_state(
#             initial_board_state, player
#         )

#         # Calculate relative improvement
#         relative_player_score = player_score - initial_player_score
#         relative_opponent_score = opponent_score - initial_opponent_score

#         # Weight absolute and relative scores
#         progress = (total_spots - empty_spots) / total_spots
#         weight_absolute = 1 - progress
#         weight_relative = progress

#         player_score = (
#             weight_absolute * player_score + weight_relative * relative_player_score
#         )
#         opponent_score = (
#             weight_absolute * opponent_score + weight_relative * relative_opponent_score
#         )

#     return player_score, opponent_score

def evaluate_board_state(board_state, player):
    """
    Evaluate the board state relative to an initial board state for better decision-making.
    
    Args:
        board_state (np.ndarray): 2D array representing the board state.
        player (int): The player for whom the evaluation is being done (1 or 2).
        initial_board_state (np.ndarray, optional): The initial board state for comparison.
        
    Returns:
        tuple: (player_score, opponent_score)
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

    # Penalize states with more empty tiles
    penalty_factor = empty_spots * 2  # Adjust this factor as needed

    # Mobility factor
    player_valid_moves = get_valid_moves(board_state, player)
    opp_valid_moves = get_valid_moves(board_state, 3 - player)

    player_mobility = len(player_valid_moves)
    opponent_mobility = len(opp_valid_moves)

    # Reward fewer valid moves (endgame preference)
    # player_score += max(0, 100 - penalty_factor) - player_mobility
    # opponent_score += max(0, 100 - penalty_factor) - opponent_mobility

    # Endgame evaluation
    if empty_spots == 0:
        is_endgame, p0_score, p1_score = check_endgame(board_state, player, 3 - player)
        if is_endgame:
            return (1, 0) if p0_score > p1_score else (0, 1)

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

            # Stable discs
            if (row, col) in corners:
                score += 10

            # Add score to the appropriate player
            if current_player == player:
                player_score += score
            else:
                opponent_score += score

    return player_score, opponent_score

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

    def mcts_search(self, player, start_time, time_limit=1.9):
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
        for move in moves:
            if move in corners:
                return move

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
                if len(node.children) > 0:
                    # Select a random child node to explore
                    node = np.random.choice(node.children)
                else:
                    # No moves to expand, proceed to backpropagation
                    p0_score, p1_score = evaluate_board_state(self.board_state, player)

                    winner = 0
                    if p0_score > p1_score:
                        winner = player
                    else:
                        winner = 3 - player

                    node.backpropagate(winner, moves_played)
                    continue

            # Simulation: Simulate a game from the current node to determine the winner
            winner, moves_played = node.rollout()

            # Backpropagation
            node.backpropagate(winner, moves_played)

        print(depth_sum / iterations)
        print(depth_sum)
        print(iterations)
        # Return the move with the most promising child
        if self.children:

            # Pure MCTS selection
            # best_move = max(self.children, key=lambda c: c.wins / (c.visits + 1e-6)).move

            # Pure RAVE selection
            best_move = max(
                self.children,
                key=lambda c: (1 - c.beta()) * (c.wins / (c.visits + 1e-6)) +
                            c.beta() * (c.rave_wins.get(c.move, 0) / (c.rave_visits.get(c.move, 0) + 1e-6))
            ).move

            # Hybrid MCTS/RAVE selection
            # best_move = max(
            #     self.children,
            #     key=lambda c: (
            #         0.35 * (c.wins / (c.visits + 1e-6)) +  # Weighted MCTS win rate
            #         0.65 * (c.rave_wins.get(c.move, 0) / (c.rave_visits.get(c.move, 0) + 1e-6))  # Weighted AMAF win rate
            #     )
            # ).move
        else:
            # If no children were expanded, return a random valid move
            if moves:
                best_move = moves[np.random.randint(len(moves))]
            else:
                best_move = None  # No valid moves, must pass
        return best_move
    
    # Expand the node by adding child nodes for each valid move
    def expand(self, corners, edges):
        if not self.children:
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
                next_player = 3 - self.current_player
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
        edges = get_board_edges(board_state)

        best_moves = []
        best_score = float('-inf')

        # Sort by importance of the move
        valid_moves.sort(key=lambda move: (
                0 if move in corners else 
                1 if move in edges else 
                2
            ))

        for move in valid_moves:
            # Always take a corner if available
            if move in corners:
                return move

            # Initialize move score
            move_score = 0

            # Heuristic: Number of opponent's pieces captured
            move_score += count_capture(board_state, move, player) * 10

            # Edge priority
            if move in edges:
                move_score += 60

            # Track the best moves
            if move_score > best_score:
                best_moves = [move]
                best_score = move_score
            elif move_score == best_score:
                best_moves.append(move)

        # Return a random move among the best-scoring moves

        return best_moves[np.random.randint(len(best_moves))] if best_moves else valid_moves[np.random.randint(len(valid_moves))]

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
        base_rollout_depth = 100  # Minimum depth at the start of the game
        max_rollout_depth = 100 # Maximum possible depth late in the game

        # Dynamically adjust max rollout depth based on game progress
        board_size = len(self.board_state) * len(self.board_state[0])  # Total board size
        empty_cells = count_zeros(self.board_state)
        progress = 1 - (empty_cells / board_size)  # Progress: 0.0 (start) to 1.0 (end)

        # Adjust rollout depth: starts at base_rollout_depth and increases as the game progresses
        adjusted_rollout_depth = min(max_rollout_depth, int(base_rollout_depth + progress * (max_rollout_depth - base_rollout_depth)))
        # print(f"Rollout depth adjusted to: {adjusted_rollout_depth} (Progress: {progress:.2f})")

        for _ in range(adjusted_rollout_depth):
            move = self.rollout_policy(current_state, player)
            if move is None:
                break  # No valid moves, game may be over
            moves_played.append((player, move))
            execute_move(current_state, move, player)
            player, opponent = opponent, player

        # Approximate winner based on current board state
        p0_score_start, p1_score_start = evaluate_board_state(self.board_state, player)
        p0_score, p1_score = evaluate_board_state(current_state, player)

        winner = 0
        if p0_score > p1_score and (p0_score - p0_score_start) >= 0:
            winner = player
        else:
            winner = 3 - player

        return winner, moves_played

    # Backpropagate the results of the simulation
    def backpropagate(self, winner, moves_played):
        node = self
        while node is not None:
            node.visits += 1
            player_who_moved = 1 if node.current_player == 2 else 2
            if player_who_moved == winner:
                # Reward shorter games by scaling wins based on remaining moves
                node.wins += 1  # Reward inversely proportional to moves played
            for player, move in moves_played:
                if move is not None:
                    node.rave_visits[move] = node.rave_visits.get(move, 0) + 1
                    if player_who_moved == winner:
                        node.rave_wins[move] = node.rave_wins.get(move, 0) + 1
            
            node.best_child_cache = None
            node.best_score_cache = None
            node = node.parent


    # Check if the node has children
    def has_children(self):
        return len(self.children) > 0

    # Calculate the beta value for RAVE
    def beta(self):
        k = 1e-5  # RAVE constant
        rave_visits = self.rave_visits.get(self.move, 0)
        return rave_visits / (rave_visits + self.visits + k)

    # Select the best child node based on the UCT/RAVE formula
    def best_child(self, exploration=np.sqrt(2)):
        # Check if cache is valid
        if self.best_child_cache is not None:
            return self.best_child_cache
        
        best_score = -np.inf
        best_child = None

        for child in self.children:
            if child.visits == 0:
                return child
            beta = self.beta()
            q_value = child.wins / child.visits
            rave_win = child.rave_wins.get(child.move, 0)
            rave_visit = child.rave_visits.get(child.move, 0)
            rave_q = rave_win / rave_visit if rave_visit > 0 else 0
            uct_value = ((1 - beta) * q_value + beta * rave_q) + exploration * math.sqrt(math.log(self.visits) / child.visits)

            if uct_value > best_score:
                best_score = uct_value
                best_child = child
        
        # Cache the best child and its score
        self.best_child_cache = best_child
        self.best_score_cache = best_score
        return best_child

