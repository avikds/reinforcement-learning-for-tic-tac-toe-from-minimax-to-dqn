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

# Step 44 - tic_tac_toe_reward
def tic_tac_toe_reward(game_status, agent_player):
    """Return scalar reward from the agent's perspective.

    game_status: one of 'X_win', 'O_win', 'draw', 'ongoing'.
    agent_player: +1 for X, -1 for O.
    """
    if game_status == 'draw' or game_status == 'ongoing':
        return 0.0

    if game_status == 'X_win':
        winner = 1
    elif game_status == 'O_win':
        winner = -1
    else:
        return 0.0

    return 1.0 if winner == agent_player else -1.0

# Step 45 - q_learning_nonterminal_target
def q_learning_nonterminal_target(
    reward,
    gamma,
    q_table,
    next_state_key,
    next_legal_actions
):
    """Return the TD target r + gamma * max_a' Q(s', a') over legal next actions."""
    if not next_legal_actions:
        return float(reward)

    max_next_q = max(
        get_q_value(q_table, next_state_key, action)
        for action in next_legal_actions
    )

    return float(reward + gamma * max_next_q)

# Step 46 - q_learning_terminal_target
def q_learning_terminal_target(reward):
    """Return the TD target for a terminal transition."""
    return float(reward)

# Step 47 - q_learning_update
def q_learning_update(q_table, state_key, action, target, alpha):
    """Apply Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a)) and return the new value."""
    current_q = get_q_value(q_table, state_key, action)
    new_q = current_q + alpha * (target - current_q)

    set_q_value(q_table, state_key, action, new_q)

    return float(new_q)

# Step 48 - episode_reset_game
def episode_reset_game():
    """Return a fresh empty board and the starting player (+1 for X)."""
    return create_empty_board(), 1

# Step 49 - episode_agent_pick_action
def episode_agent_pick_action(q_table, board, current_player, epsilon, rng):
    """Return (canonical_state_key, action_index_0_to_8) using epsilon-greedy over legal moves."""
    state_key = canonical_board_key(board)

    legal_moves = get_legal_moves(board)
    legal_actions = [row * 3 + col for row, col in legal_moves]

    action = epsilon_greedy_select_action(
        q_table,
        state_key,
        legal_actions,
        epsilon,
        rng
    )

    return state_key, int(action)

# Step 50 - episode_apply_action
def episode_apply_action(board, action, current_player, agent_player):
    """Apply one move, return next_board/next_player/status/reward/done."""
    row = action // 3
    col = action % 3

    next_board = place_move(board, row, col, current_player)
    status = get_game_status(next_board)
    reward = tic_tac_toe_reward(status, agent_player)
    done = status != 'ongoing'

    next_player = switch_player(current_player)

    return {
        'next_board': next_board,
        'next_player': next_player,
        'status': status,
        'reward': float(reward),
        'done': done
    }

# Step 51 - episode_apply_q_update
def episode_apply_q_update(
    q_table,
    state_key,
    action,
    reward,
    next_board,
    done,
    alpha,
    gamma
):
    """Compute the TD target (terminal or nonterminal) and apply the Q-learning update."""
    if done:
        target = q_learning_terminal_target(reward)
    else:
        next_state_key = canonical_board_key(next_board)
        next_legal_actions = get_legal_moves(next_board)

        target = q_learning_nonterminal_target(
            reward,
            gamma,
            q_table,
            next_state_key,
            next_legal_actions
        )

    return q_learning_update(
        q_table,
        state_key,
        action,
        target,
        alpha
    )

# Step 52 - episode_check_terminate
def episode_check_terminate(status):
    """Return True if status is terminal (win or draw), else False."""
    return status != 'ongoing'

# Step 53 - train_q_learning_agent
def train_q_learning_agent(
    num_episodes,
    alpha,
    gamma,
    initial_epsilon,
    min_epsilon,
    decay_rate,
    opponent_policy,
    rng
):
    """Run Q-learning episodes with the agent playing X and return the trained table."""
    q_table = initialize_q_table()
    episode_outcomes = []

    for episode_index in range(num_episodes):
        epsilon = epsilon_decay_schedule(
            initial_epsilon,
            episode_index,
            min_epsilon,
            decay_rate
        )

        board, current_player = episode_reset_game()
        agent_player = 1
        status = 'ongoing'

        while not episode_check_terminate(status):
            # Agent (X) chooses an action.
            state_key, action = episode_agent_pick_action(
                q_table,
                board,
                current_player,
                epsilon,
                rng
            )

            transition = episode_apply_action(
                board,
                action,
                current_player,
                agent_player
            )

            next_board = transition['next_board']
            next_player = transition['next_player']
            status = transition['status']

            # If the agent itself ended the game, update immediately.
            if transition['done']:
                episode_apply_q_update(
                    q_table,
                    state_key,
                    action,
                    transition['reward'],
                    next_board,
                    True,
                    alpha,
                    gamma
                )
                board = next_board
                break

            # Opponent (O) responds.
            opponent_action = opponent_policy(
                next_board,
                next_player,
                rng
            )

            opponent_transition = episode_apply_action(
                next_board,
                opponent_action,
                next_player,
                agent_player
            )

            board = opponent_transition['next_board']
            current_player = opponent_transition['next_player']
            status = opponent_transition['status']

            # Update the agent's decision using the state after the opponent response.
            episode_apply_q_update(
                q_table,
                state_key,
                action,
                opponent_transition['reward'],
                board,
                opponent_transition['done'],
                alpha,
                gamma
            )

            if opponent_transition['done']:
                break

        episode_outcomes.append(status)

    return {
        'q_table': q_table,
        'episode_outcomes': episode_outcomes
    }

# Step 54 - compute_batched_outcome_stats
def compute_batched_outcome_stats(episode_outcomes, batch_size):
    """Aggregate outcomes into per-batch win/loss/draw rates."""
    num_complete_batches = len(episode_outcomes) // batch_size

    batch_indices = []
    win_rates = []
    loss_rates = []
    draw_rates = []

    for batch_index in range(num_complete_batches):
        start = batch_index * batch_size
        end = start + batch_size
        batch = episode_outcomes[start:end]

        batch_indices.append(batch_index)
        win_rates.append(batch.count('win') / batch_size)
        loss_rates.append(batch.count('loss') / batch_size)
        draw_rates.append(batch.count('draw') / batch_size)

    return {
        'batch_index': np.array(batch_indices, dtype=int),
        'win_rate': np.array(win_rates, dtype=float),
        'loss_rate': np.array(loss_rates, dtype=float),
        'draw_rate': np.array(draw_rates, dtype=float)
    }

# Step 55 - self_play_episode
def self_play_episode(q_table, alpha, gamma, epsilon, rng):
    """Run one self-play episode and return final_status and a list of transitions."""
    board, current_player = episode_reset_game()
    transitions = []
    status = 'ongoing'

    while not episode_check_terminate(status):
        state_key, action = episode_agent_pick_action(
            q_table,
            board,
            current_player,
            epsilon,
            rng
        )

        transition = episode_apply_action(
            board,
            action,
            current_player,
            current_player
        )

        transitions.append({
            'state_key': state_key,
            'action': action,
            'reward': transition['reward'],
            'next_board': transition['next_board'],
            'done': transition['done'],
            'player': current_player
        })

        board = transition['next_board']
        current_player = transition['next_player']
        status = transition['status']

    return {
        'final_status': status,
        'transitions': transitions
    }

# Step 56 - flip_board_perspective
def flip_board_perspective(board, current_player):
    """Return a board view where current_player's marks are +1."""
    return (board * current_player).astype(int, copy=True)

# Step 57 - perspective_reward_sign
def perspective_reward_sign(reward, acting_player, scoring_player):
    """Return reward expressed from acting_player's perspective."""
    if acting_player == scoring_player:
        return float(reward)

    return float(-reward)

# Step 58 - train_q_agent_self_play
def train_q_agent_self_play(
    num_episodes,
    alpha,
    gamma,
    initial_epsilon,
    min_epsilon,
    decay_rate,
    rng
):
    """Run self-play Q-learning with a shared Q-table for both players."""
    q_table = initialize_q_table()
    episode_outcomes = []

    for episode_index in range(num_episodes):
        epsilon = epsilon_decay_schedule(
            initial_epsilon,
            episode_index,
            min_epsilon,
            decay_rate
        )

        episode = self_play_episode(
            q_table,
            alpha,
            gamma,
            epsilon,
            rng
        )

        final_status = episode['final_status']
        episode_outcomes.append(final_status)

        for transition in episode['transitions']:
            player = transition['player']
            state_key = transition['state_key']
            action = transition['action']
            next_board = transition['next_board']
            done = transition['done']
            reward = transition['reward']

            # Express the state from the acting player's perspective.
            # The stored action is still the flat 0..8 cell index.
            # Recompute the canonical key from the perspective-flipped board.
            # This ensures X and O share the same Q-table representation.
            #
            # The original transition state is represented by state_key,
            # but to apply the perspective-aware update we reconstruct the
            # board state from the transition's next_board and played action.
            if player == 1:
                perspective_state_key = state_key
                scoring_player = 1 if final_status == 'X_win' else -1
            else:
                scoring_player = 1 if final_status == 'X_win' else -1

                # Recover the pre-action board from next_board.
                pre_board = next_board.copy()
                row = action // 3
                col = action % 3
                pre_board[row, col] = 0

                perspective_board = flip_board_perspective(
                    pre_board,
                    player
                )
                perspective_state_key = canonical_board_key(
                    perspective_board
                )

            acting_reward = perspective_reward_sign(
                reward,
                player,
                scoring_player
            )

            perspective_next_board = flip_board_perspective(
                next_board,
                player
            )

            if done:
                target = q_learning_terminal_target(acting_reward)
            else:
                next_state_key = canonical_board_key(
                    perspective_next_board
                )
                next_legal_moves = get_legal_moves(
                    perspective_next_board
                )

                target = q_learning_nonterminal_target(
                    acting_reward,
                    gamma,
                    q_table,
                    next_state_key,
                    next_legal_moves
                )

            q_learning_update(
                q_table,
                perspective_state_key,
                action,
                target,
                alpha
            )

    return {
        'q_table': q_table,
        'episode_outcomes': episode_outcomes
    }

# Step 59 - evaluate_q_agent_vs_random
def evaluate_q_agent_vs_random(q_table, num_games, rng):
    """Play num_games between the greedy Q-agent and a random opponent.

    Returns a dict with keys 'wins', 'losses', 'draws' (ints) and
    'win_rate', 'loss_rate', 'draw_rate' (floats), all from the agent's
    perspective. The agent alternates between playing X and O across games.
    """
    wins = 0
    losses = 0
    draws = 0

    if num_games == 0:
        return {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rate': 0.0,
            'loss_rate': 0.0,
            'draw_rate': 0.0
        }

    for game_index in range(num_games):
        board = create_empty_board()
        agent_player = 1 if game_index % 2 == 0 else -1
        current_player = 1

        while True:
            if current_player == agent_player:
                perspective_board = flip_board_perspective(
                    board,
                    agent_player
                )
                state_key = canonical_board_key(perspective_board)

                legal_moves = get_legal_moves(perspective_board)
                legal_actions = [
                    row * 3 + col
                    for row, col in legal_moves
                ]

                action = greedy_argmax_over_legal_actions(
                    q_table,
                    state_key,
                    legal_actions,
                    rng
                )

            else:
                action = random_move_agent(
                    board,
                    current_player,
                    rng
                )
                action = action[0] * 3 + action[1]

            row = action // 3
            col = action % 3
            board = place_move(board, row, col, current_player)

            status = get_game_status(board)

            if status != 'ongoing':
                reward = tic_tac_toe_reward(status, agent_player)

                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1

                break

            current_player = switch_player(current_player)

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': wins / num_games,
        'loss_rate': losses / num_games,
        'draw_rate': draws / num_games
    }

# Step 60 - evaluate_q_agent_vs_minimax
def evaluate_q_agent_vs_minimax(q_table, num_games, rng):
    """Evaluate the greedy Q-agent against an optimal minimax opponent."""
    outcomes = []

    for game_index in range(num_games):
        board = create_empty_board()
        agent_player = 1 if game_index % 2 == 0 else -1
        current_player = 1

        while True:
            if current_player == agent_player:
                perspective_board = flip_board_perspective(
                    board,
                    agent_player
                )
                state_key = canonical_board_key(perspective_board)

                legal_moves = get_legal_moves(perspective_board)
                legal_actions = [
                    row * 3 + col
                    for row, col in legal_moves
                ]

                action = greedy_argmax_over_legal_actions(
                    q_table,
                    state_key,
                    legal_actions,
                    rng
                )

                row = action // 3
                col = action % 3

            else:
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
                reward = tic_tac_toe_reward(status, agent_player)

                if reward > 0:
                    outcomes.append('X_win')
                elif reward < 0:
                    outcomes.append('O_win')
                else:
                    outcomes.append('draw')

                break

            current_player = switch_player(current_player)

    return compute_outcome_rates(outcomes)

# Step 61 - inspect_q_values_for_state
def inspect_q_values_for_state(q_table, board, current_player):
    """Print the board and Q-values for all 9 cells; return a length-9 array."""
    state_key = canonical_board_key(board)

    values = np.array([
        get_q_value(q_table, state_key, (row, col))
        for row in range(3)
        for col in range(3)
    ], dtype=float)

    print_board(board)

    for row in range(3):
        start = row * 3
        print(' '.join(f'{value:+.2f}' for value in values[start:start + 3]))

    return values

# Step 62 - serialize_q_table_to_dict
def serialize_q_table_to_dict(q_table):
    """Convert a Q-table (str -> np.ndarray shape (9,)) into a plain dict (str -> list of floats)."""
    return {
        str(state_key): [float(value) for value in values]
        for state_key, values in q_table.items()
    }

# Step 63 - deserialize_q_table_from_dict
def deserialize_q_table_from_dict(serialized):
    """Rebuild a Q-table (state_key -> np.ndarray shape (9,)) from a plain dict."""
    return {
        str(state_key): np.asarray(values, dtype=np.float64)
        for state_key, values in serialized.items()
    }

# Step 64 - encode_board_flat_length_nine
def encode_board_flat_length_nine(board, current_player):
    """Encode a 3x3 board as a length-9 float32 vector from current_player's view."""
    return (board * current_player).astype(np.float32).reshape(9)

# Step 65 - encode_board_one_hot_length_eighteen
def encode_board_one_hot_length_eighteen(board, current_player):
    """Encode a 3x3 board as a length-18 two-channel one-hot vector."""
    perspective_board = board * current_player

    own_pieces = (perspective_board == 1).astype(np.float32).reshape(9)
    opponent_pieces = (perspective_board == -1).astype(np.float32).reshape(9)

    return np.concatenate([own_pieces, opponent_pieces]).astype(np.float32)

# Step 66 - build_mlp_architecture
def build_mlp_architecture(input_dim, hidden_dim, output_dim=9):
    """Return a dict describing the MLP layer dimensions."""
    return {
        'input_dim': int(input_dim),
        'hidden_dim': int(hidden_dim),
        'output_dim': int(output_dim)
    }

# Step 67 - initialize_mlp_parameters
def initialize_mlp_parameters(architecture, seed=0):
    """Initialize MLP weights with He init and zero biases.

    architecture: dict from build_mlp_architecture with input_dim, hidden_dim, output_dim.
    seed: int seed for numpy RNG.
    Returns dict with keys 'W1', 'b1', 'W2', 'b2'.
    """
    np.random.seed(seed)

    input_dim = architecture['input_dim']
    hidden_dim = architecture['hidden_dim']
    output_dim = architecture['output_dim']

    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden_dim, dtype=float)

    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(output_dim, dtype=float)

    return {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

# Step 68 - mlp_forward_pass
def mlp_forward_pass(params, x):
    """Forward pass through a two-layer MLP with ReLU hidden activation.

    Args:
        params: dict with keys 'W1', 'b1', 'W2', 'b2'.
        x: np.ndarray of shape (batch, input_dim).

    Returns:
        (q_values, cache) where q_values has shape (batch, output_dim) and
        cache is a dict with keys {'x', 'z1', 'h1', 'q'}.
    """
    z1 = x @ params['W1'] + params['b1']
    h1 = np.maximum(0.0, z1)
    q = h1 @ params['W2'] + params['b2']

    cache = {
        'x': x,
        'z1': z1,
        'h1': h1,
        'q': q
    }

    return q, cache

# Step 69 - mask_illegal_actions_neg_inf
def mask_illegal_actions_neg_inf(q_values, legal_action_mask):
    """Return a copy of q_values with illegal entries set to -inf."""
    masked_q_values = np.array(q_values, dtype=float, copy=True)
    masked_q_values[~legal_action_mask] = -np.inf
    return masked_q_values

# Step 70 - argmax_action_from_q_values
def argmax_action_from_q_values(masked_q_values):
    """Return the index of the largest entry in masked_q_values as an int."""
    return int(np.argmax(masked_q_values))

# Step 71 - mse_loss_on_chosen_action
def mse_loss_on_chosen_action(predicted_q, action_indices, target_q):
    """MSE between Q(s, a_taken) and the bootstrapped target Q."""
    chosen_q = predicted_q[
        np.arange(predicted_q.shape[0]),
        action_indices
    ]

    return float(np.mean((chosen_q - target_q) ** 2))

# Step 72 - mlp_backward_pass
def mlp_backward_pass(params, cache, action_indices, target_q):
    """Backprop MSE-on-chosen-action loss through the MLP and return param gradients."""
    x = cache['x']
    z1 = cache['z1']
    h1 = cache['h1']
    q = cache['q']

    batch_size = x.shape[0]

    # Gradient of the loss with respect to the output Q-values.
    dq = np.zeros_like(q, dtype=float)
    rows = np.arange(batch_size)
    chosen_q = q[rows, action_indices]

    dq[rows, action_indices] = 2.0 * (chosen_q - target_q) / batch_size

    # Output layer gradients.
    dW2 = h1.T @ dq
    db2 = np.sum(dq, axis=0)

    # Backpropagate through ReLU.
    dh1 = dq @ params['W2'].T
    dz1 = dh1 * (z1 > 0)

    # Hidden layer gradients.
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2
    }

# Step 73 - adam_update_step
def adam_update_step(
    params,
    grads,
    adam_state,
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8
):
    """Perform one Adam optimizer step and return updated params and state."""
    if 't' not in adam_state:
        adam_state['t'] = 0

    if 'm' not in adam_state:
        adam_state['m'] = {}

    if 'v' not in adam_state:
        adam_state['v'] = {}

    adam_state['t'] += 1
    t = adam_state['t']

    new_params = {}

    for key in params:
        if key not in adam_state['m']:
            adam_state['m'][key] = np.zeros_like(params[key], dtype=float)
            adam_state['v'][key] = np.zeros_like(params[key], dtype=float)

        adam_state['m'][key] = (
            beta1 * adam_state['m'][key]
            + (1.0 - beta1) * grads[key]
        )

        adam_state['v'][key] = (
            beta2 * adam_state['v'][key]
            + (1.0 - beta2) * (grads[key] ** 2)
        )

        m_hat = adam_state['m'][key] / (1.0 - beta1 ** t)
        v_hat = adam_state['v'][key] / (1.0 - beta2 ** t)

        new_params[key] = (
            params[key]
            - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        )

    return new_params, adam_state

# Step 74 - create_replay_buffer
from collections import deque

def create_replay_buffer(capacity):
    """Return an empty replay buffer with a fixed maximum capacity."""
    return {
        'data': deque(maxlen=capacity),
        'capacity': int(capacity)
    }

# Step 75 - append_transition_to_buffer
def append_transition_to_buffer(
    buffer,
    state,
    action,
    reward,
    next_state,
    done,
    next_legal_mask
):
    """Append one (s, a, r, s', done, next_legal_mask) transition to the replay buffer."""
    buffer['data'].append(
        (
            state,
            action,
            reward,
            next_state,
            done,
            next_legal_mask
        )
    )

    return buffer

# Step 76 - cap_buffer_size_drop_oldest
def cap_buffer_size_drop_oldest(buffer):
    """Drop oldest transitions until len(buffer['data']) <= buffer['capacity']."""
    while len(buffer['data']) > buffer['capacity']:
        if hasattr(buffer['data'], 'popleft'):
            buffer['data'].popleft()
        else:
            buffer['data'].pop(0)

    return buffer

# Step 77 - sample_minibatch_from_buffer
def sample_minibatch_from_buffer(buffer, batch_size, rng):
    """Draw `batch_size` random transitions from `buffer` and stack fields into arrays."""
    data = buffer['data']

    indices = rng.integers(0, len(data), size=batch_size)
    transitions = [data[int(i)] for i in indices]

    return {
        'states': np.asarray(
            [transition['state'] for transition in transitions]
        ),
        'actions': np.asarray(
            [transition['action'] for transition in transitions],
            dtype=int
        ),
        'rewards': np.asarray(
            [transition['reward'] for transition in transitions],
            dtype=float
        ),
        'next_states': np.asarray(
            [transition['next_state'] for transition in transitions]
        ),
        'dones': np.asarray(
            [transition['done'] for transition in transitions],
            dtype=bool
        ),
        'next_legal_masks': np.asarray(
            [transition['next_legal_mask'] for transition in transitions],
            dtype=bool
        )
    }

# Step 78 - build_target_network_copy
def build_target_network_copy(online_params):
    """Return a deep copy of the online MLP parameter dict."""
    return {
        key: value.copy()
        for key, value in online_params.items()
    }

# Step 79 - compute_target_q_with_target_network
def compute_target_q_with_target_network(target_params, batch, gamma):
    """Compute DQN bootstrap targets r + gamma * max_a' Q_target(s', a')."""
    next_states = batch['next_states']
    rewards = np.asarray(batch['rewards'], dtype=float)
    dones = np.asarray(batch['dones'], dtype=bool)
    next_legal_masks = np.asarray(batch['next_legal_masks'], dtype=bool)

    next_q_values, _ = mlp_forward_pass(target_params, next_states)

    masked_next_q_values = mask_illegal_actions_neg_inf(
        next_q_values,
        next_legal_masks
    )

    max_next_q_values = np.max(masked_next_q_values, axis=1)

    targets = rewards.copy()
    non_terminal = ~dones
    targets[non_terminal] += (
        gamma * max_next_q_values[non_terminal]
    )

    return targets

# Step 80 - sync_target_network_periodically
def sync_target_network_periodically(
    online_params,
    target_params,
    step_count,
    sync_every_k
):
    """Copy online -> target every sync_every_k steps; otherwise leave target unchanged."""
    if step_count > 0 and step_count % sync_every_k == 0:
        return build_target_network_copy(online_params)

    return target_params

# Step 81 - dqn_select_action
def dqn_select_action(online_params, state, legal_mask, epsilon, rng):
    """Epsilon-greedy action index over the legal moves."""
    if rng.random() < epsilon:
        legal_actions = np.flatnonzero(legal_mask)
        index = rng.integers(len(legal_actions))
        return int(legal_actions[index])

    state = np.asarray(state)

    if state.ndim == 1:
        network_input = state.reshape(1, -1)
    else:
        network_input = state

    q_values, _ = mlp_forward_pass(online_params, network_input)
    q_values = q_values[0]

    masked_q_values = mask_illegal_actions_neg_inf(
        q_values,
        np.asarray(legal_mask, dtype=bool)
    )

    return argmax_action_from_q_values(masked_q_values)

# Step 82 - dqn_train_step
def dqn_train_step(
    online_params,
    target_params,
    adam_state,
    buffer,
    batch_size,
    gamma,
    lr,
    rng
):
    """Run one DQN minibatch update. Return (online_params, adam_state, loss)."""
    batch = sample_minibatch_from_buffer(
        buffer,
        batch_size,
        rng
    )

    target_q = compute_target_q_with_target_network(
        target_params,
        batch,
        gamma
    )

    predicted_q, cache = mlp_forward_pass(
        online_params,
        batch['states']
    )

    loss = mse_loss_on_chosen_action(
        predicted_q,
        batch['actions'],
        target_q
    )

    grads = mlp_backward_pass(
        online_params,
        cache,
        batch['actions'],
        target_q
    )

    online_params, adam_state = adam_update_step(
        online_params,
        grads,
        adam_state,
        learning_rate=lr
    )

    return online_params, adam_state, float(loss)

# Step 83 - train_dqn_agent
def train_dqn_agent(
    num_episodes,
    hidden_dim=64,
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    buffer_capacity=10000,
    sync_every_k=200,
    epsilon_start=1.0,
    epsilon_end=0.05,
    seed=0
):
    """Full DQN self-play training loop. Returns dict with online_params,
    target_params, loss_history, reward_history, architecture.
    """
    architecture = build_mlp_architecture(
        input_dim=9,
        hidden_dim=hidden_dim,
        output_dim=9
    )

    online_params = initialize_mlp_parameters(
        architecture,
        seed=seed
    )

    target_params = build_target_network_copy(online_params)
    adam_state = {}
    buffer = create_replay_buffer(buffer_capacity)
    rng = np.random.default_rng(seed)

    loss_history = []
    reward_history = []

    total_steps = 0

    for episode_index in range(num_episodes):
        # Linear epsilon annealing from epsilon_start to epsilon_end.
        if num_episodes <= 1:
            epsilon = float(epsilon_end)
        else:
            fraction = episode_index / (num_episodes - 1)
            epsilon = float(
                epsilon_start
                + fraction * (epsilon_end - epsilon_start)
            )

        board, current_player = episode_reset_game()
        status = 'ongoing'
        episode_reward = 0.0
        episode_losses = []

        while not episode_check_terminate(status):
            # Current state from the perspective of the player to move.
            state = encode_board_flat_length_nine(
                board,
                current_player
            )

            legal_mask = np.zeros(9, dtype=bool)

            for row, col in get_legal_moves(board):
                legal_mask[row * 3 + col] = True

            # Epsilon-greedy action.
            action = dqn_select_action(
                online_params,
                state,
                legal_mask,
                epsilon,
                rng
            )

            row = action // 3
            col = action % 3

            # Apply move.
            next_board = place_move(
                board,
                row,
                col,
                current_player
            )

            status = get_game_status(next_board)
            done = episode_check_terminate(status)

            # Reward is from the perspective of the player who acted.
            reward = tic_tac_toe_reward(
                status,
                current_player
            )

            episode_reward += float(reward)

            next_player = switch_player(current_player)

            # Encode next state from the next player's perspective.
            next_state = encode_board_flat_length_nine(
                next_board,
                next_player
            )

            # Legal-action mask for the next state.
            next_legal_mask = np.zeros(9, dtype=bool)

            if not done:
                for next_row, next_col in get_legal_moves(next_board):
                    next_legal_mask[next_row * 3 + next_col] = True

            # IMPORTANT:
            # Store a dictionary because Step 77's sampler expects
            # transition['state'], transition['action'], etc.
            buffer['data'].append({
                'state': state,
                'action': int(action),
                'reward': float(reward),
                'next_state': next_state,
                'done': bool(done),
                'next_legal_mask': next_legal_mask
            })

            # Explicitly enforce the configured capacity.
            cap_buffer_size_drop_oldest(buffer)

            total_steps += 1

            # Train once enough samples are available.
            if len(buffer['data']) >= batch_size:
                online_params, adam_state, loss = dqn_train_step(
                    online_params,
                    target_params,
                    adam_state,
                    buffer,
                    batch_size,
                    gamma,
                    lr,
                    rng
                )

                episode_losses.append(float(loss))

            # Periodically synchronize target network.
            target_params = sync_target_network_periodically(
                online_params,
                target_params,
                total_steps,
                sync_every_k
            )

            board = next_board
            current_player = next_player

            if done:
                break

        # One history entry per episode.
        reward_history.append(float(episode_reward))

        if episode_losses:
            loss_history.append(float(np.mean(episode_losses)))
        else:
            loss_history.append(0.0)

    return {
        'online_params': online_params,
        'target_params': target_params,
        'loss_history': loss_history,
        'reward_history': reward_history,
        'architecture': architecture
    }

# Step 84 - compare_dqn_tabular_random_minimax
def compare_dqn_tabular_random_minimax(dqn_artifacts, q_table, num_games=200):
    """Round-robin evaluation among DQN, tabular Q, random, and minimax agents."""
    rng = np.random.default_rng(0)

    dqn_params = dqn_artifacts['online_params']

    def choose_action(agent_name, board, player):
        if agent_name == 'random':
            return random_move_agent(board, player, rng)

        if agent_name == 'minimax':
            _, move = minimax_alpha_beta(
                board,
                player,
                -float('inf'),
                float('inf')
            )
            return move

        # Both learned agents represent the position from the
        # current player's perspective.
        perspective_board = flip_board_perspective(board, player)
        state_key = canonical_board_key(perspective_board)

        legal_moves = get_legal_moves(board)
        legal_actions = [
            row * 3 + col
            for row, col in legal_moves
        ]

        if agent_name == 'dqn':
            state = encode_board_flat_length_nine(
                board,
                player
            )

            legal_mask = np.zeros(9, dtype=bool)
            legal_mask[legal_actions] = True

            action = dqn_select_action(
                dqn_params,
                state,
                legal_mask,
                0.0,
                rng
            )
            return int(action // 3), int(action % 3)

        # Tabular Q agent.
        action = greedy_argmax_over_legal_actions(
            q_table,
            state_key,
            legal_actions,
            rng
        )
        return int(action // 3), int(action % 3)

    def play_match(first_agent, second_agent):
        wins = 0
        draws = 0
        losses = 0

        for game_index in range(num_games):
            board = create_empty_board()

            # Alternate which agent gets X.
            first_agent_player = 1 if game_index % 2 == 0 else -1
            second_agent_player = -first_agent_player

            current_player = 1

            while True:
                if current_player == first_agent_player:
                    move = choose_action(
                        first_agent,
                        board,
                        current_player
                    )
                else:
                    move = choose_action(
                        second_agent,
                        board,
                        current_player
                    )

                row, col = move
                board = place_move(
                    board,
                    row,
                    col,
                    current_player
                )

                status = get_game_status(board)

                if status != 'ongoing':
                    if status == 'draw':
                        draws += 1
                    elif (
                        (status == 'X_win' and first_agent_player == 1)
                        or
                        (status == 'O_win' and first_agent_player == -1)
                    ):
                        wins += 1
                    else:
                        losses += 1

                    break

                current_player = switch_player(current_player)

        if num_games == 0:
            return {
                'wins': 0.0,
                'draws': 0.0,
                'losses': 0.0
            }

        return {
            'wins': wins / num_games,
            'draws': draws / num_games,
            'losses': losses / num_games
        }

    matchups = [
        ('dqn_vs_random', 'dqn', 'random'),
        ('dqn_vs_minimax', 'dqn', 'minimax'),
        ('dqn_vs_tabular', 'dqn', 'tabular'),
        ('tabular_vs_random', 'tabular', 'random'),
        ('tabular_vs_minimax', 'tabular', 'minimax'),
        ('random_vs_minimax', 'random', 'minimax')
    ]

    return {
        key: play_match(first_agent, second_agent)
        for key, first_agent, second_agent in matchups
    }

# Step 85 - sarsa_on_policy_update
def sarsa_on_policy_update(
    q_table,
    state_key,
    action,
    reward,
    next_state_key,
    next_action,
    done,
    alpha,
    gamma
):
    """Apply one on-policy SARSA update and return the updated q_table."""
    current_q = get_q_value(q_table, state_key, action)

    if done:
        target = float(reward)
    else:
        next_q = get_q_value(q_table, next_state_key, next_action)
        target = float(reward + gamma * next_q)

    new_q = current_q + alpha * (target - current_q)

    set_q_value(
        q_table,
        state_key,
        action,
        new_q
    )

    return q_table

# Step 86 - train_sarsa_agent
def train_sarsa_agent(
    num_episodes,
    alpha,
    gamma,
    initial_epsilon,
    min_epsilon,
    decay_rate,
    opponent_policy,
    rng
):
    """Run num_episodes of on-policy SARSA vs opponent_policy."""
    q_table = initialize_q_table()
    episode_outcomes = []

    for episode_index in range(num_episodes):
        epsilon = epsilon_decay_schedule(
            initial_epsilon,
            episode_index,
            min_epsilon,
            decay_rate
        )

        board, current_player = episode_reset_game()
        status = 'ongoing'

        # Pending agent transition awaiting the next agent action.
        pending_state_key = None
        pending_action = None
        pending_reward = 0.0

        while not episode_check_terminate(status):
            # Agent is always X and current_player should be 1 here.
            state_key, action = episode_agent_pick_action(
                q_table,
                board,
                current_player,
                epsilon,
                rng
            )

            agent_transition = episode_apply_action(
                board,
                action,
                current_player,
                1
            )

            next_board = agent_transition['next_board']
            status = agent_transition['status']
            agent_reward = agent_transition['reward']
            agent_done = agent_transition['done']

            if agent_done:
                # Terminal agent move: no bootstrap.
                sarsa_on_policy_update(
                    q_table,
                    state_key,
                    action,
                    agent_reward,
                    state_key,
                    action,
                    True,
                    alpha,
                    gamma
                )

                board = next_board
                break

            # Opponent O responds.
            next_player = agent_transition['next_player']
            opponent_action = opponent_policy(
                next_board,
                next_player,
                rng
            )

            opponent_transition = episode_apply_action(
                next_board,
                opponent_action,
                next_player,
                1
            )

            board_after_opponent = opponent_transition['next_board']
            status = opponent_transition['status']
            opponent_reward = opponent_transition['reward']
            opponent_done = opponent_transition['done']

            if opponent_done:
                # Opponent ended the game. The previous agent action
                # gets the terminal reward from the agent's perspective.
                sarsa_on_policy_update(
                    q_table,
                    state_key,
                    action,
                    opponent_reward,
                    state_key,
                    action,
                    True,
                    alpha,
                    gamma
                )

                board = board_after_opponent
                break

            # The agent is about to act again. Select the next on-policy
            # action before updating the previous state-action pair.
            next_state_key, next_action = episode_agent_pick_action(
                q_table,
                board_after_opponent,
                1,
                epsilon,
                rng
            )

            # Update the previous agent transition using the action
            # actually selected by the current epsilon-greedy policy.
            sarsa_on_policy_update(
                q_table,
                state_key,
                action,
                opponent_reward,
                next_state_key,
                next_action,
                False,
                alpha,
                gamma
            )

            board = board_after_opponent

            # The next agent action has already been selected, so play it
            # directly on the board.
            agent_transition = episode_apply_action(
                board,
                next_action,
                1,
                1
            )

            board = agent_transition['next_board']
            status = agent_transition['status']

            if agent_transition['done']:
                sarsa_on_policy_update(
                    q_table,
                    next_state_key,
                    next_action,
                    agent_transition['reward'],
                    next_state_key,
                    next_action,
                    True,
                    alpha,
                    gamma
                )
                break

            # Continue from the opponent's response to this new agent move.
            current_state_key = next_state_key
            current_action = next_action
            current_reward = agent_transition['reward']

            opponent_action = opponent_policy(
                board,
                -1,
                rng
            )

            opponent_transition = episode_apply_action(
                board,
                opponent_action,
                -1,
                1
            )

            board = opponent_transition['next_board']
            status = opponent_transition['status']

            if opponent_transition['done']:
                sarsa_on_policy_update(
                    q_table,
                    current_state_key,
                    current_action,
                    opponent_transition['reward'],
                    current_state_key,
                    current_action,
                    True,
                    alpha,
                    gamma
                )
                break

            next_state_key, next_action = episode_agent_pick_action(
                q_table,
                board,
                1,
                epsilon,
                rng
            )

            sarsa_on_policy_update(
                q_table,
                current_state_key,
                current_action,
                opponent_transition['reward'],
                next_state_key,
                next_action,
                False,
                alpha,
                gamma
            )

            # The loop now needs to continue from this newly selected
            # agent action. Rather than duplicating the transition logic
            # indefinitely, restart the main cycle with the selected move.
            while not episode_check_terminate(status):
                agent_transition = episode_apply_action(
                    board,
                    next_action,
                    1,
                    1
                )

                board = agent_transition['next_board']
                status = agent_transition['status']

                if agent_transition['done']:
                    sarsa_on_policy_update(
                        q_table,
                        next_state_key,
                        next_action,
                        agent_transition['reward'],
                        next_state_key,
                        next_action,
                        True,
                        alpha,
                        gamma
                    )
                    break

                opponent_action = opponent_policy(
                    board,
                    -1,
                    rng
                )

                opponent_transition = episode_apply_action(
                    board,
                    opponent_action,
                    -1,
                    1
                )

                board = opponent_transition['next_board']
                status = opponent_transition['status']

                if opponent_transition['done']:
                    sarsa_on_policy_update(
                        q_table,
                        next_state_key,
                        next_action,
                        opponent_transition['reward'],
                        next_state_key,
                        next_action,
                        True,
                        alpha,
                        gamma
                    )
                    break

                new_state_key, new_action = episode_agent_pick_action(
                    q_table,
                    board,
                    1,
                    epsilon,
                    rng
                )

                sarsa_on_policy_update(
                    q_table,
                    next_state_key,
                    next_action,
                    opponent_transition['reward'],
                    new_state_key,
                    new_action,
                    False,
                    alpha,
                    gamma
                )

                next_state_key = new_state_key
                next_action = new_action

    # Record the final outcome for the episode.
        episode_outcomes.append(status)

    return {
        'q_table': q_table,
        'episode_outcomes': episode_outcomes
    }

# Step 87 - reinforce_log_prob_of_action
def reinforce_log_prob_of_action(logits, legal_action_mask, action):
    """Return (log_prob_of_action, full_prob_vector) under a masked softmax policy."""
    masked_logits = mask_illegal_actions_neg_inf(
        logits,
        legal_action_mask
    )

    max_logit = np.max(masked_logits)
    shifted_logits = masked_logits - max_logit

    exp_logits = np.exp(shifted_logits)
    exp_logits[~legal_action_mask] = 0.0

    probs = exp_logits / np.sum(exp_logits)

    log_prob = float(np.log(probs[action]))

    return log_prob, probs

# Step 88 - reinforce_collect_episode_returns
def reinforce_collect_episode_returns(rewards, gamma):
    """Return discounted returns G_t for a REINFORCE episode as a numpy array of shape (T,)."""
    returns = np.zeros(len(rewards), dtype=float)
    
    running_return = 0.0

    for t in range(len(rewards) - 1, -1, -1):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns

# Step 89 - reinforce_policy_gradient_update
def reinforce_policy_gradient_update(
    params,
    episode_cache,
    returns,
    adam_state,
    learning_rate=1e-2
):
    """Apply one REINFORCE update that ascends sum_t G_t log pi(a_t|s_t)."""
    states = np.asarray(episode_cache['states'])
    actions = np.asarray(episode_cache['actions'], dtype=int)
    legal_masks = np.asarray(episode_cache['legal_masks'], dtype=bool)
    returns = np.asarray(returns, dtype=float)

    # Forward pass through the policy network.
    logits, cache = mlp_forward_pass(params, states)

    batch_size = states.shape[0]

    # Stable masked softmax.
    masked_logits = mask_illegal_actions_neg_inf(
        logits,
        legal_masks
    )

    max_logits = np.max(masked_logits, axis=1, keepdims=True)
    shifted_logits = masked_logits - max_logits

    exp_logits = np.exp(shifted_logits)
    exp_logits[~legal_masks] = 0.0

    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Gradient of log pi(a|s):
    # d log pi / d logits = one_hot(action) - pi.
    dlogits = -returns[:, None] * probs

    rows = np.arange(batch_size)
    dlogits[rows, actions] += returns

    # REINFORCE maximizes the objective, while Adam performs gradient
    # descent. Therefore negate the policy-gradient objective here.
    dq = -dlogits

    h1 = cache['h1']
    z1 = cache['z1']
    x = cache['x']

    # Output layer.
    dW2 = h1.T @ dq
    db2 = np.sum(dq, axis=0)

    # ReLU backpropagation.
    dh1 = dq @ params['W2'].T
    dz1 = dh1 * (z1 > 0)

    # Hidden layer.
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)

    grads = {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2
    }

    return adam_update_step(
        params,
        grads,
        adam_state,
        learning_rate=learning_rate
    )

# Step 90 - train_reinforce_agent
def train_reinforce_agent(
    num_episodes,
    gamma,
    learning_rate,
    hidden_dim,
    opponent_policy,
    rng,
    init_seed=0
):
    """Train a policy MLP with REINFORCE against the supplied opponent."""
    architecture = build_mlp_architecture(9, hidden_dim, 9)

    params = initialize_mlp_parameters(
        architecture,
        seed=init_seed
    )

    adam_state = {}
    episode_outcomes = []

    for _ in range(num_episodes):
        board, current_player = episode_reset_game()

        states = []
        actions = []
        legal_masks = []
        rewards = []

        status = 'ongoing'

        while not episode_check_terminate(status):
            if current_player == 1:
                # Agent's turn.
                state = encode_board_flat_length_nine(
                    board,
                    current_player
                )

                legal_mask = np.zeros(9, dtype=bool)
                for row, col in get_legal_moves(board):
                    legal_mask[row * 3 + col] = True

                logits, _ = mlp_forward_pass(
                    params,
                    state.reshape(1, -1)
                )
                logits = logits[0]

                masked_logits = mask_illegal_actions_neg_inf(
                    logits,
                    legal_mask
                )

                max_logit = np.max(masked_logits)
                exp_logits = np.exp(masked_logits - max_logit)
                exp_logits[~legal_mask] = 0.0
                probs = exp_logits / np.sum(exp_logits)

                action = int(
                    rng.choice(
                        np.arange(9),
                        p=probs
                    )
                )

                row = action // 3
                col = action % 3

                next_board = place_move(
                    board,
                    row,
                    col,
                    current_player
                )

                next_status = get_game_status(next_board)
                done = next_status != 'ongoing'

                # The agent receives reward only on its final move.
                reward = tic_tac_toe_reward(
                    next_status,
                    1
                ) if done else 0.0

                states.append(state)
                actions.append(action)
                legal_masks.append(legal_mask)
                rewards.append(float(reward))

                board = next_board
                status = next_status

                if done:
                    break

                current_player = switch_player(current_player)

            else:
                # Opponent's turn.
                action = opponent_policy(
                    board,
                    current_player,
                    rng
                )

                # Allow opponent policies to return either a flat index
                # or a (row, col) tuple.
                if isinstance(action, tuple):
                    row, col = action
                else:
                    row = action // 3
                    col = action % 3

                board = place_move(
                    board,
                    row,
                    col,
                    current_player
                )

                status = get_game_status(board)

                if status != 'ongoing':
                    break

                current_player = switch_player(current_player)

        episode_outcomes.append(status)

        if states:
            episode_cache = {
                'states': np.asarray(states, dtype=np.float32),
                'actions': np.asarray(actions, dtype=int),
                'legal_masks': np.asarray(legal_masks, dtype=bool),
                'forward_caches': []
            }

            returns = reinforce_collect_episode_returns(
                rewards,
                gamma
            )

            params, adam_state = reinforce_policy_gradient_update(
                params,
                episode_cache,
                returns,
                adam_state,
                learning_rate=learning_rate
            )

    return {
        'params': params,
        'architecture': architecture,
        'episode_outcomes': episode_outcomes
    }

# Step 91 - compare_value_vs_policy_learners
def compare_value_vs_policy_learners(num_episodes=5000, eval_games=200, seed=0):
    """Train Q-learning, SARSA, and REINFORCE under matched settings."""
    # Shared training settings.
    alpha = 0.5
    gamma = 0.95
    initial_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.001

    # Random opponent policy returning a flat action index.
    def random_opponent_policy(board, player, rng):
        row, col = random_move_agent(board, player, rng)
        return row * 3 + col

    # Keep independent RNG streams so each learner has reproducible
    # behavior without one learner consuming another's random sequence.
    q_rng = np.random.default_rng(seed)
    sarsa_rng = np.random.default_rng(seed)
    reinforce_rng = np.random.default_rng(seed)

    # Train tabular Q-learning.
    q_result = train_q_learning_agent(
        num_episodes,
        alpha,
        gamma,
        initial_epsilon,
        min_epsilon,
        decay_rate,
        random_opponent_policy,
        q_rng
    )

    # Train SARSA.
    sarsa_result = train_sarsa_agent(
        num_episodes,
        alpha,
        gamma,
        initial_epsilon,
        min_epsilon,
        decay_rate,
        random_opponent_policy,
        sarsa_rng
    )

    # Train REINFORCE.
    reinforce_result = train_reinforce_agent(
        num_episodes,
        gamma,
        1e-2,
        32,
        random_opponent_policy,
        reinforce_rng,
        init_seed=seed
    )

    def outcomes_to_learning_curve(outcomes):
        scores = []

        for outcome in outcomes:
            if outcome == 'X_win':
                scores.append(1.0)
            elif outcome == 'O_win':
                scores.append(-1.0)
            else:
                scores.append(0.0)

        return scores

    def evaluate_tabular(q_table, eval_seed):
        rng = np.random.default_rng(eval_seed)

        if eval_games == 0:
            return {
                'win_rate_vs_random': 0.0,
                'draw_rate_vs_minimax': 0.0
            }

        random_result = evaluate_q_agent_vs_random(
            q_table,
            eval_games,
            rng
        )

        minimax_result = evaluate_q_agent_vs_minimax(
            q_table,
            eval_games,
            rng
        )

        return {
            'win_rate_vs_random': float(random_result['win_rate']),
            'draw_rate_vs_minimax': float(minimax_result['draw_rate'])
        }

    def evaluate_reinforce(params, eval_seed):
        rng = np.random.default_rng(eval_seed)

        if eval_games == 0:
            return {
                'win_rate_vs_random': 0.0,
                'draw_rate_vs_minimax': 0.0
            }

        random_wins = 0
        random_draws = 0
        random_losses = 0

        minimax_wins = 0
        minimax_draws = 0
        minimax_losses = 0

        for game_index in range(eval_games):
            agent_player = 1 if game_index % 2 == 0 else -1

            def choose_policy_action(board, player):
                state = encode_board_flat_length_nine(
                    board,
                    player
                )

                legal_mask = np.zeros(9, dtype=bool)
                for row, col in get_legal_moves(board):
                    legal_mask[row * 3 + col] = True

                logits, _ = mlp_forward_pass(
                    params,
                    state.reshape(1, -1)
                )

                masked_logits = mask_illegal_actions_neg_inf(
                    logits[0],
                    legal_mask
                )

                return argmax_action_from_q_values(masked_logits)

            # REINFORCE vs random.
            board = create_empty_board()
            current_player = 1

            while True:
                if current_player == agent_player:
                    action = choose_policy_action(
                        board,
                        current_player
                    )
                    row, col = action // 3, action % 3
                else:
                    row, col = random_move_agent(
                        board,
                        current_player,
                        rng
                    )

                board = place_move(
                    board,
                    row,
                    col,
                    current_player
                )

                status = get_game_status(board)

                if status != 'ongoing':
                    reward = tic_tac_toe_reward(
                        status,
                        agent_player
                    )

                    if reward > 0:
                        random_wins += 1
                    elif reward < 0:
                        random_losses += 1
                    else:
                        random_draws += 1
                    break

                current_player = switch_player(current_player)

            # REINFORCE vs minimax.
            board = create_empty_board()
            current_player = 1

            while True:
                if current_player == agent_player:
                    action = choose_policy_action(
                        board,
                        current_player
                    )
                    row, col = action // 3, action % 3
                else:
                    _, move = minimax_alpha_beta(
                        board,
                        current_player,
                        -float('inf'),
                        float('inf')
                    )
                    row, col = move

                board = place_move(
                    board,
                    row,
                    col,
                    current_player
                )

                status = get_game_status(board)

                if status != 'ongoing':
                    reward = tic_tac_toe_reward(
                        status,
                        agent_player
                    )

                    if reward > 0:
                        minimax_wins += 1
                    elif reward < 0:
                        minimax_losses += 1
                    else:
                        minimax_draws += 1
                    break

                current_player = switch_player(current_player)

        return {
            'win_rate_vs_random': random_wins / eval_games,
            'draw_rate_vs_minimax': minimax_draws / eval_games
        }

    q_eval = evaluate_tabular(
        q_result['q_table'],
        seed + 100
    )

    sarsa_eval = evaluate_tabular(
        sarsa_result['q_table'],
        seed + 200
    )

    reinforce_eval = evaluate_reinforce(
        reinforce_result['params'],
        seed + 300
    )

    return {
        'q_learning': {
            'win_rate_vs_random': q_eval['win_rate_vs_random'],
            'draw_rate_vs_minimax': q_eval['draw_rate_vs_minimax'],
            'learning_curve': outcomes_to_learning_curve(
                q_result['episode_outcomes']
            )
        },
        'sarsa': {
            'win_rate_vs_random': sarsa_eval['win_rate_vs_random'],
            'draw_rate_vs_minimax': sarsa_eval['draw_rate_vs_minimax'],
            'learning_curve': outcomes_to_learning_curve(
                sarsa_result['episode_outcomes']
            )
        },
        'reinforce': {
            'win_rate_vs_random': reinforce_eval['win_rate_vs_random'],
            'draw_rate_vs_minimax': reinforce_eval['draw_rate_vs_minimax'],
            'learning_curve': outcomes_to_learning_curve(
                reinforce_result['episode_outcomes']
            )
        }
    }

# Step 92 - symmetry_augmented_training
def symmetry_augmented_training(
    q_table,
    state_board,
    action,
    reward,
    next_state_board,
    done,
    alpha,
    gamma
):
    """Apply Q-learning updates to all 8 D4 symmetries of a transition."""
    action_row = action // 3
    action_col = action % 3

    # Generate the 8 D4 symmetries:
    # 4 rotations, with and without horizontal reflection.
    for k in range(4):
        rotated_state = np.rot90(state_board, k)
        rotated_next_state = np.rot90(next_state_board, k)

        candidates = [
            (rotated_state, rotated_next_state, False),
            (
                np.fliplr(rotated_state),
                np.fliplr(rotated_next_state),
                True
            )
        ]

        for transformed_state, transformed_next_state, reflected in candidates:
            # Transform the action using the same geometric operation.
            action_marker = np.zeros((3, 3), dtype=int)
            action_marker[action_row, action_col] = 1

            transformed_marker = np.rot90(action_marker, k)

            if reflected:
                transformed_marker = np.fliplr(transformed_marker)

            transformed_action_row, transformed_action_col = np.argwhere(
                transformed_marker == 1
            )[0]

            transformed_action = (
                int(transformed_action_row) * 3
                + int(transformed_action_col)
            )

            # Use the transformed state directly as the Q-table state key.
            state_key = encode_board_state_key(transformed_state)

            if done:
                target = q_learning_terminal_target(reward)
            else:
                next_state_key = encode_board_state_key(
                    transformed_next_state
                )

                next_legal_moves = get_legal_moves(
                    transformed_next_state
                )

                next_legal_actions = [
                    row * 3 + col
                    for row, col in next_legal_moves
                ]

                target = q_learning_nonterminal_target(
                    reward,
                    gamma,
                    q_table,
                    next_state_key,
                    next_legal_actions
                )

            q_learning_update(
                q_table,
                state_key,
                transformed_action,
                target,
                alpha
            )

    return q_table

