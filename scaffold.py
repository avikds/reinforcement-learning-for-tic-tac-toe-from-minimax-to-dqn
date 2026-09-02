"""
Reinforcement Learning for Tic-Tac-Toe: From Minimax to DQN scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Scaffold for the Tic-Tac-Toe RL lab: from minimax to DQN.

Imports every helper the student implements in solution.py and runs a
minimal end-to-end demo (board ops -> minimax -> tabular Q -> DQN).
"""


import numpy as np

from solution import (
    create_empty_board,
    encode_player,
    print_board,
    is_cell_empty,
    place_move,
    get_legal_moves,
    check_row_win,
    check_column_win,
    check_main_diagonal_win,
    check_anti_diagonal_win,
    is_winner,
    is_draw,
    get_game_status,
    get_current_player,
    switch_player,
    play_hardcoded_game,
    play_interactive_game,
    TicTacToeGame,
    random_move_agent,
    play_random_vs_random_game,
    play_random_vs_random_matches,
    compute_outcome_rates,
    minimax_terminal_score,
    minimax_value,
    minimax_recursive,
    minimax_max_min_step,
    minimax_best_move,
    minimax_alpha_beta,
    play_minimax_vs_random_matches,
    play_minimax_vs_minimax_matches,
    encode_board_state_key,
    canonical_board_key,
    initialize_q_table,
    get_q_value,
    set_q_value,
    choose_learning_rate_alpha,
    choose_discount_factor_gamma,
    choose_initial_epsilon,
    epsilon_decay_schedule,
    epsilon_greedy_explore_move,
    epsilon_greedy_select_action,
    greedy_argmax_over_legal_actions,
    random_tie_break_argmax,
    tic_tac_toe_reward,
    q_learning_nonterminal_target,
    q_learning_terminal_target,
    q_learning_update,
    episode_reset_game,
    episode_agent_pick_action,
    episode_apply_action,
    episode_apply_q_update,
    episode_check_terminate,
    train_q_learning_agent,
    compute_batched_outcome_stats,
    self_play_episode,
    flip_board_perspective,
    perspective_reward_sign,
    train_q_agent_self_play,
    evaluate_q_agent_vs_random,
    evaluate_q_agent_vs_minimax,
    inspect_q_values_for_state,
    serialize_q_table_to_dict,
    deserialize_q_table_from_dict,
    encode_board_flat_length_nine,
    encode_board_one_hot_length_eighteen,
    build_mlp_architecture,
    initialize_mlp_parameters,
    mlp_forward_pass,
    mask_illegal_actions_neg_inf,
    argmax_action_from_q_values,
    mse_loss_on_chosen_action,
    mlp_backward_pass,
    adam_update_step,
    create_replay_buffer,
    append_transition_to_buffer,
    cap_buffer_size_drop_oldest,
    sample_minibatch_from_buffer,
    build_target_network_copy,
    compute_target_q_with_target_network,
    sync_target_network_periodically,
    dqn_select_action,
    dqn_train_step,
    train_dqn_agent,
    compare_dqn_tabular_random_minimax,
    sarsa_on_policy_update,
    train_sarsa_agent,
    reinforce_log_prob_of_action,
    reinforce_collect_episode_returns,
    reinforce_policy_gradient_update,
    train_reinforce_agent,
    compare_value_vs_policy_learners,
    symmetry_augmented_training,
)


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # 1) Game engine sanity check.
    board = create_empty_board()
    print("Empty board:")
    print_board(board)
    board = place_move(board, 1, 1, encode_player("X"))
    board = place_move(board, 0, 0, encode_player("O"))
    print("After two moves:")
    print_board(board)
    print("Status:", get_game_status(board))
    print("Legal moves remaining:", len(get_legal_moves(board)))

    # 2) Random vs random baseline.
    outcomes = play_random_vs_random_matches(200, rng)
    print("Random-vs-random rates:", compute_outcome_rates(outcomes))

    # 3) Minimax must never lose to a random opponent.
    mm_outcomes = play_minimax_vs_random_matches(20, minimax_plays_x=True, rng=rng)
    print("Minimax(X) vs Random rates:", compute_outcome_rates(mm_outcomes))

    # 4) Train a tabular Q-learning agent against random play.
    q_table = train_q_learning_agent(
        num_episodes=2000,
        alpha=choose_learning_rate_alpha(),
        gamma=choose_discount_factor_gamma(),
        initial_epsilon=choose_initial_epsilon(),
        min_epsilon=0.05,
        decay_rate=1e-3,
        opponent_policy=random_move_agent,
        rng=rng,
    )
    q_vs_random = evaluate_q_agent_vs_random(q_table, 100, rng)
    print("Tabular Q vs Random:", q_vs_random)

    # 5) Quick DQN smoke test (few episodes just to verify the loop runs).
    dqn_artifacts = train_dqn_agent(
        num_episodes=50,
        hidden_dim=32,
        gamma=0.95,
        lr=1e-3,
        batch_size=16,
        buffer_capacity=500,
        sync_every_k=50,
        epsilon_start=1.0,
        epsilon_end=0.1,
        seed=0,
    )
    print("DQN training artifacts keys:", list(dqn_artifacts.keys()) if isinstance(dqn_artifacts, dict) else type(dqn_artifacts).__name__)

    # 6) DQN forward-pass demonstration on a fresh board.
    demo_board = create_empty_board()
    x_vec = encode_board_flat_length_nine(demo_board, encode_player("X"))
    arch = build_mlp_architecture(input_dim=9, hidden_dim=32, output_dim=9)
    params = initialize_mlp_parameters(arch, seed=0)
    q_values, _ = mlp_forward_pass(params, x_vec.reshape(1, -1))
    legal_mask = np.ones(9, dtype=bool)
    masked = mask_illegal_actions_neg_inf(q_values, legal_mask)
    action = argmax_action_from_q_values(masked)
    print("Untrained MLP picks action index:", int(np.asarray(action).ravel()[0]))

    print("Scaffold demo complete.")
