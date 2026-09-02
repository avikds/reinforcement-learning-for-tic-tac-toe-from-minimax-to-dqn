"""
Reinforcement Learning for Tic-Tac-Toe: From Minimax to DQN

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - create_empty_board
import numpy as np

def create_empty_board():
    """Return an empty 3x3 Tic-Tac-Toe board as an int numpy array of zeros."""
    return np.zeros((3, 3), dtype=int)

# Step 2 - encode_player
def encode_player(player):
    """Return the integer encoding for 'X', 'O', or 'empty'."""
    mapping = {
        'X': 1,
        'O': -1,
        'empty': 0
    }
    return mapping[player]

# Step 3 - print_board
def print_board(board):
    """Print the 3x3 board using X, O, and . characters."""
    symbols = {
        1: 'X',
        -1: 'O',
        0: '.'
    }

    for row in board:
        print(' '.join(symbols[cell] for cell in row))

# Step 4 - is_cell_empty
def is_cell_empty(board, row, col):
    """Return True if board[row, col] is empty (0), else False."""
    return bool(board[row, col] == 0)

# Step 5 - place_move
def place_move(board, row, col, player):
    """Place player's mark at (row, col) and return the new board."""
    if not is_cell_empty(board, row, col):
        raise ValueError("Cell is already occupied.")

    new_board = board.copy()
    new_board[row, col] = player
    return new_board

# Step 6 - get_legal_moves
def get_legal_moves(board):
    """Return a list of (row, col) tuples for all empty cells on the board."""
    legal_moves = []

    for row in range(3):
        for col in range(3):
            if board[row, col] == 0:
                legal_moves.append((row, col))

    return legal_moves

# Step 7 - check_row_win
def check_row_win(board, player):
    """Return True if `player` has three-in-a-row across any row of `board`."""
    return bool(np.any(np.all(board == player, axis=1)))

# Step 8 - check_column_win
def check_column_win(board, player):
    """Return True if `player` has three-in-a-row in any column of `board`."""
    return bool(np.any(np.all(board == player, axis=0)))

# Step 9 - check_main_diagonal_win
def check_main_diagonal_win(board, player):
    """Return True if `player` occupies all three main-diagonal cells."""
    return bool(np.all(np.diag(board) == player))

# Step 10 - check_anti_diagonal_win
def check_anti_diagonal_win(board, player):
    """Return True if `player` occupies all three anti-diagonal cells."""
    return bool(np.all(np.fliplr(board).diagonal() == player))

# Step 11 - is_winner
def is_winner(board, player):
    """Return True if `player` has three-in-a-row on `board`."""
    return bool(
        check_row_win(board, player)
        or check_column_win(board, player)
        or check_main_diagonal_win(board, player)
        or check_anti_diagonal_win(board, player)
    )

# Step 12 - is_draw
def is_draw(board):
    """Return True iff the board is full and neither player has won."""
    return bool(
        not get_legal_moves(board)
        and not is_winner(board, 1)
        and not is_winner(board, -1)
    )

# Step 13 - get_game_status
def get_game_status(board):
    """Return 'X_win', 'O_win', 'draw', or 'ongoing' for the given 3x3 board."""
    if is_winner(board, 1):
        return 'X_win'
    elif is_winner(board, -1):
        return 'O_win'
    elif is_draw(board):
        return 'draw'
    else:
        return 'ongoing'

# Step 14 - get_current_player
def get_current_player(board):
    """Return 1 if X is to move, -1 if O is to move."""
    x_count = np.count_nonzero(board == 1)
    o_count = np.count_nonzero(board == -1)

    return 1 if x_count == o_count else -1

# Step 15 - switch_player
def switch_player(player):
    """Return the opponent of `player` (1 <-> -1)."""
    return -player

# Step 16 - play_hardcoded_game
def play_hardcoded_game(moves):
    """Replay a fixed sequence of (row, col) moves and return (final_board, status)."""
    board = create_empty_board()
    current_player = 1

    for row, col in moves:
        board = place_move(board, row, col, current_player)

        status = get_game_status(board)
        if status != 'ongoing':
            return board, status

        current_player = switch_player(current_player)

    return board, get_game_status(board)

# Step 17 - play_interactive_game
def play_interactive_game():
    """Play a full game with two humans entering moves via stdin and return the final status."""
    board = create_empty_board()
    current_player = 1

    while True:
        print_board(board)

        row, col = map(int, input().split())

        if not is_cell_empty(board, row, col):
            continue

        board = place_move(board, row, col, current_player)

        status = get_game_status(board)

        if status != 'ongoing':
            print_board(board)
            return status

        current_player = switch_player(current_player)

# Step 18 - TicTacToeGame
class TicTacToeGame:
    """Stateful Tic-Tac-Toe environment wrapping the Part 1 engine."""

    def __init__(self):
        self.board = create_empty_board()
        self.current_player = 1
        self.status = 'ongoing'

    def reset(self):
        """Return the game to its empty starting state."""
        self.board = create_empty_board()
        self.current_player = 1
        self.status = 'ongoing'
        return self.board

    def legal_moves(self):
        """Return a list of currently legal moves."""
        return get_legal_moves(self.board)

    def is_terminal(self):
        """Return True once the game has ended."""
        return self.status != 'ongoing'

    def step(self, row, col):
        """Play the current player's move and update the game state."""
        if self.is_terminal():
            raise ValueError("Game is already over.")

        self.board = place_move(self.board, row, col, self.current_player)
        self.status = get_game_status(self.board)

        if self.status == 'ongoing':
            self.current_player = switch_player(self.current_player)

        return self.board, self.status

# Step 19 - random_move_agent
def random_move_agent(board, player, rng):
    """Return a uniformly random legal (row, col) move for `player`."""
    legal_moves = get_legal_moves(board)
    index = rng.integers(len(legal_moves))
    row, col = legal_moves[index]
    return int(row), int(col)

# Step 20 - play_random_vs_random_game
def play_random_vs_random_game(rng):
    """Simulate one full random-vs-random game and return the final status."""
    board = create_empty_board()
    current_player = 1

    while True:
        row, col = random_move_agent(board, current_player, rng)
        board = place_move(board, row, col, current_player)

        status = get_game_status(board)
        if status != 'ongoing':
            return status

        current_player = switch_player(current_player)

# Step 21 - play_random_vs_random_matches
def play_random_vs_random_matches(n_games, rng):
    """Run n_games random-vs-random games and return the list of outcome strings."""
    outcomes = []

    for _ in range(n_games):
        outcomes.append(play_random_vs_random_game(rng))

    return outcomes

# Step 22 - compute_outcome_rates
def compute_outcome_rates(outcomes):
    """Return {'x_win_rate','o_win_rate','draw_rate'} from a list of outcome labels."""
    total_games = len(outcomes)

    if total_games == 0:
        return {
            'x_win_rate': 0.0,
            'o_win_rate': 0.0,
            'draw_rate': 0.0
        }

    return {
        'x_win_rate': outcomes.count('X_win') / total_games,
        'o_win_rate': outcomes.count('O_win') / total_games,
        'draw_rate': outcomes.count('draw') / total_games
    }

# Step 23 - minimax_terminal_score
def minimax_terminal_score(status):
    """Return +1 for 'X_win', -1 for 'O_win', 0 for 'draw'."""
    if status == 'X_win':
        return 1
    elif status == 'O_win':
        return -1
    else:
        return 0

# Step 24 - minimax_value
def minimax_value(board, player):
    """Return the minimax value of `board` with `player` to move."""
    status = get_game_status(board)

    if status != 'ongoing':
        return minimax_terminal_score(status)

    legal_moves = get_legal_moves(board)
    child_values = []

    for row, col in legal_moves:
        next_board = place_move(board, row, col, player)
        value = minimax_value(next_board, switch_player(player))
        child_values.append(value)

    if player == 1:
        return max(child_values)
    else:
        return min(child_values)

# Step 25 - minimax_recursive
def minimax_recursive(board, player):
    """Return the memoized minimax value of `board` with `player` to move."""
    if not hasattr(minimax_recursive, "_cache"):
        minimax_recursive._cache = {}

    cache = minimax_recursive._cache
    key = (board.tobytes(), player)

    if key in cache:
        return cache[key]

    status = get_game_status(board)

    if status != 'ongoing':
        value = minimax_terminal_score(status)
        cache[key] = value
        return value

    legal_moves = get_legal_moves(board)
    child_values = []

    for row, col in legal_moves:
        next_board = place_move(board, row, col, player)
        value = minimax_recursive(next_board, switch_player(player))
        child_values.append(value)

    if player == 1:
        value = max(child_values)
    else:
        value = min(child_values)

    cache[key] = value
    return value

# Step 26 - minimax_max_min_step
def minimax_max_min_step(board, player):
    """Return (best_score, best_move) after expanding one minimax level."""
    legal_moves = get_legal_moves(board)

    if player == 1:
        best_score = -float("inf")
    else:
        best_score = float("inf")

    best_move = None

    for row, col in legal_moves:
        next_board = place_move(board, row, col, player)
        score = minimax_recursive(next_board, switch_player(player))

        if player == 1:
            if score > best_score:
                best_score = score
                best_move = (row, col)
        else:
            if score < best_score:
                best_score = score
                best_move = (row, col)

    return best_score, best_move

# Step 27 - minimax_best_move
def minimax_best_move(board, player):
    """Return the optimal (row, col) move for `player` via minimax."""
    _, best_move = minimax_max_min_step(board, player)
    return int(best_move[0]), int(best_move[1])

# Step 28 - minimax_alpha_beta
def minimax_alpha_beta(board, player, alpha, beta):
    """Return (best_score, best_move) for `player` using alpha-beta pruning."""
    status = get_game_status(board)

    if status != 'ongoing':
        return minimax_terminal_score(status), None

    legal_moves = get_legal_moves(board)
    best_move = legal_moves[0]

    if player == 1:
        best_score = -float("inf")

        for row, col in legal_moves:
            next_board = place_move(board, row, col, player)
            score, _ = minimax_alpha_beta(
                next_board,
                switch_player(player),
                alpha,
                beta
            )

            if score > best_score:
                best_score = score
                best_move = (row, col)

            alpha = max(alpha, best_score)

            if alpha >= beta:
                break

    else:
        best_score = float("inf")

        for row, col in legal_moves:
            next_board = place_move(board, row, col, player)
            score, _ = minimax_alpha_beta(
                next_board,
                switch_player(player),
                alpha,
                beta
            )

            if score < best_score:
                best_score = score
                best_move = (row, col)

            beta = min(beta, best_score)

            if alpha >= beta:
                break

    return best_score, best_move

# Step 29 - play_minimax_vs_random_matches
def play_minimax_vs_random_matches(n_games, minimax_plays_x, rng):
    """Run n_games of minimax vs random and return aggregated outcome rates."""
    outcomes = []

    for _ in range(n_games):
        board = create_empty_board()
        current_player = 1

        while True:
            if current_player == 1:
                if minimax_plays_x:
                    move = minimax_best_move(board, current_player)
                else:
                    move = random_move_agent(board, current_player, rng)
            else:
                if minimax_plays_x:
                    move = random_move_agent(board, current_player, rng)
                else:
                    move = minimax_best_move(board, current_player)

            row, col = move
            board = place_move(board, row, col, current_player)

            status = get_game_status(board)
            if status != 'ongoing':
                outcomes.append(status)
                break

            current_player = switch_player(current_player)

    return compute_outcome_rates(outcomes)

# Step 30 - play_minimax_vs_minimax_matches
def play_minimax_vs_minimax_matches(n_games):
    """Play n_games minimax-vs-minimax games and report outcome rates plus an all_draws flag."""
    outcomes = []

    for _ in range(n_games):
        board = create_empty_board()
        current_player = 1

        while True:
            _, move = minimax_alpha_beta(
                board,
                current_player,
                -float("inf"),
                float("inf")
            )

            row, col = move
            board = place_move(board, row, col, current_player)

            status = get_game_status(board)
            if status != 'ongoing':
                outcomes.append(status)
                break

            current_player = switch_player(current_player)

    rates = compute_outcome_rates(outcomes)
    rates['all_draws'] = all(status == 'draw' for status in outcomes)

    return rates

# Step 31 - encode_board_state_key
def encode_board_state_key(board):
    """Encode a 3x3 board as a length-9 string over {'0','1','2'} in row-major order."""
    mapping = {
        0: '0',
        1: '1',
        -1: '2'
    }

    return ''.join(mapping[int(cell)] for cell in board.flat)

# Step 32 - canonical_board_key
def canonical_board_key(board):
    """Return the lexicographically smallest key across all 8 board symmetries."""
    candidates = []

    for k in range(4):
        rotated = np.rot90(board, k)
        candidates.append(encode_board_state_key(rotated))
        candidates.append(encode_board_state_key(np.fliplr(rotated)))

    return min(candidates)

# Step 33 - initialize_q_table
from collections import defaultdict

def initialize_q_table():
    """Create an empty Q-table that returns 0.0 for unseen (state, action) keys."""
    return defaultdict(float)

# Step 34 - get_q_value
def get_q_value(q_table, state_key, action):
    """Return Q(state_key, action), or 0.0 if the pair is not in the table."""
    return q_table.get((state_key, action), 0.0)

# Step 35 - set_q_value
def set_q_value(q_table, state_key, action, value):
    """Write a new Q-value for a (state, action) pair into the Q-table."""
    q_table[(state_key, action)] = float(value)

# Step 36 - choose_learning_rate_alpha
def choose_learning_rate_alpha():
    """Return the learning rate alpha (float in (0, 1]) for tabular Q-learning."""
    return 0.1

# Step 37 - choose_discount_factor_gamma
def choose_discount_factor_gamma():
    """Return the discount factor gamma in [0, 1] for Q-learning."""
    return 0.9

# Step 38 - choose_initial_epsilon
def choose_initial_epsilon():
    """Return the starting exploration rate epsilon for epsilon-greedy."""
    return 1.0

# Step 39 - epsilon_decay_schedule
def epsilon_decay_schedule(initial_epsilon, episode_index, min_epsilon, decay_rate):
    """Return the decayed epsilon for the given episode, clipped to min_epsilon."""
    epsilon = initial_epsilon * np.exp(-decay_rate * episode_index)
    return float(max(min_epsilon, epsilon))

# Step 40 - epsilon_greedy_explore_move
def epsilon_greedy_explore_move(legal_actions, rng):
    """Sample a uniformly random legal action from legal_actions using rng."""
    index = rng.integers(len(legal_actions))
    return legal_actions[index]

# Step 41 - epsilon_greedy_select_action
def epsilon_greedy_select_action(q_table, state_key, legal_actions, epsilon, rng):
    """Choose an action via epsilon-greedy over the legal actions."""
    if rng.random() < epsilon:
        return epsilon_greedy_explore_move(legal_actions, rng)

    return greedy_argmax_over_legal_actions(
        q_table,
        state_key,
        legal_actions,
        rng
    )

# Step 42 - greedy_argmax_over_legal_actions
def greedy_argmax_over_legal_actions(q_table, state_key, legal_actions, rng):
    """Return the legal action with the highest Q-value (random tie-break)."""
    q_values = [
        get_q_value(q_table, state_key, action)
        for action in legal_actions
    ]

    max_q = max(q_values)

    best_actions = [
        action
        for action, q_value in zip(legal_actions, q_values)
        if q_value == max_q
    ]

    index = rng.integers(len(best_actions))
    return best_actions[index]

# Step 43 - random_tie_break_argmax
def random_tie_break_argmax(values, candidates, rng):
    """Return one candidate whose value equals max(values), tie-broken uniformly at random."""
    max_value = max(values)

    best_candidates = [
        candidate
        for value, candidate in zip(values, candidates)
        if value == max_value
    ]

    index = rng.integers(len(best_candidates))
    return best_candidates[index]

# Step 44 - tic_tac_toe_reward (not yet solved)
# TODO: implement

# Step 45 - q_learning_nonterminal_target (not yet solved)
# TODO: implement

# Step 46 - q_learning_terminal_target (not yet solved)
# TODO: implement

# Step 47 - q_learning_update (not yet solved)
# TODO: implement

# Step 48 - episode_reset_game (not yet solved)
# TODO: implement

# Step 49 - episode_agent_pick_action (not yet solved)
# TODO: implement

# Step 50 - episode_apply_action (not yet solved)
# TODO: implement

# Step 51 - episode_apply_q_update (not yet solved)
# TODO: implement

# Step 52 - episode_check_terminate (not yet solved)
# TODO: implement

# Step 53 - train_q_learning_agent (not yet solved)
# TODO: implement

# Step 54 - compute_batched_outcome_stats (not yet solved)
# TODO: implement

# Step 55 - self_play_episode (not yet solved)
# TODO: implement

# Step 56 - flip_board_perspective (not yet solved)
# TODO: implement

# Step 57 - perspective_reward_sign (not yet solved)
# TODO: implement

# Step 58 - train_q_agent_self_play (not yet solved)
# TODO: implement

# Step 59 - evaluate_q_agent_vs_random (not yet solved)
# TODO: implement

# Step 60 - evaluate_q_agent_vs_minimax (not yet solved)
# TODO: implement

# Step 61 - inspect_q_values_for_state (not yet solved)
# TODO: implement

# Step 62 - serialize_q_table_to_dict (not yet solved)
# TODO: implement

# Step 63 - deserialize_q_table_from_dict (not yet solved)
# TODO: implement

# Step 64 - encode_board_flat_length_nine (not yet solved)
# TODO: implement

# Step 65 - encode_board_one_hot_length_eighteen (not yet solved)
# TODO: implement

# Step 66 - build_mlp_architecture (not yet solved)
# TODO: implement

# Step 67 - initialize_mlp_parameters (not yet solved)
# TODO: implement

# Step 68 - mlp_forward_pass (not yet solved)
# TODO: implement

# Step 69 - mask_illegal_actions_neg_inf (not yet solved)
# TODO: implement

# Step 70 - argmax_action_from_q_values (not yet solved)
# TODO: implement

# Step 71 - mse_loss_on_chosen_action (not yet solved)
# TODO: implement

# Step 72 - mlp_backward_pass (not yet solved)
# TODO: implement

# Step 73 - adam_update_step (not yet solved)
# TODO: implement

# Step 74 - create_replay_buffer (not yet solved)
# TODO: implement

# Step 75 - append_transition_to_buffer (not yet solved)
# TODO: implement

# Step 76 - cap_buffer_size_drop_oldest (not yet solved)
# TODO: implement

# Step 77 - sample_minibatch_from_buffer (not yet solved)
# TODO: implement

# Step 78 - build_target_network_copy (not yet solved)
# TODO: implement

# Step 79 - compute_target_q_with_target_network (not yet solved)
# TODO: implement

# Step 80 - sync_target_network_periodically (not yet solved)
# TODO: implement

# Step 81 - dqn_select_action (not yet solved)
# TODO: implement

# Step 82 - dqn_train_step (not yet solved)
# TODO: implement

# Step 83 - train_dqn_agent (not yet solved)
# TODO: implement

# Step 84 - compare_dqn_tabular_random_minimax (not yet solved)
# TODO: implement

# Step 85 - sarsa_on_policy_update (not yet solved)
# TODO: implement

# Step 86 - train_sarsa_agent (not yet solved)
# TODO: implement

# Step 87 - reinforce_log_prob_of_action (not yet solved)
# TODO: implement

# Step 88 - reinforce_collect_episode_returns (not yet solved)
# TODO: implement

# Step 89 - reinforce_policy_gradient_update (not yet solved)
# TODO: implement

# Step 90 - train_reinforce_agent (not yet solved)
# TODO: implement

# Step 91 - compare_value_vs_policy_learners (not yet solved)
# TODO: implement

# Step 92 - symmetry_augmented_training (not yet solved)
# TODO: implement

