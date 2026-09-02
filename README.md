# Reinforcement Learning for Tic-Tac-Toe: From Minimax to DQN

Build a complete Tic-Tac-Toe reinforcement learning lab from scratch: the game engine, minimax baselines, tabular Q-learning with self-play, and a deep Q-network agent, then compare value-based and policy-based learners in a fully observable game you can debug end to end. A hands-on tour of exploration, bootstrapping, and function approximation.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** create_empty_board
- [x] **2.** encode_player
- [x] **3.** print_board
- [x] **4.** is_cell_empty
- [x] **5.** place_move
- [x] **6.** get_legal_moves
- [x] **7.** check_row_win
- [x] **8.** check_column_win
- [x] **9.** check_main_diagonal_win
- [x] **10.** check_anti_diagonal_win
- [x] **11.** is_winner
- [x] **12.** is_draw
- [x] **13.** get_game_status
- [x] **14.** get_current_player
- [x] **15.** switch_player
- [x] **16.** play_hardcoded_game
- [x] **17.** play_interactive_game
- [x] **18.** TicTacToeGame
- [x] **19.** random_move_agent
- [x] **20.** play_random_vs_random_game
- [x] **21.** play_random_vs_random_matches
- [x] **22.** compute_outcome_rates
- [x] **23.** minimax_terminal_score
- [x] **24.** minimax_value
- [x] **25.** minimax_recursive
- [x] **26.** minimax_max_min_step
- [x] **27.** minimax_best_move
- [x] **28.** minimax_alpha_beta
- [x] **29.** play_minimax_vs_random_matches
- [x] **30.** play_minimax_vs_minimax_matches
- [x] **31.** encode_board_state_key
- [x] **32.** canonical_board_key
- [x] **33.** initialize_q_table
- [x] **34.** get_q_value
- [x] **35.** set_q_value
- [x] **36.** choose_learning_rate_alpha
- [x] **37.** choose_discount_factor_gamma
- [x] **38.** choose_initial_epsilon
- [x] **39.** epsilon_decay_schedule
- [x] **40.** epsilon_greedy_explore_move
- [x] **41.** epsilon_greedy_select_action
- [x] **42.** greedy_argmax_over_legal_actions
- [x] **43.** random_tie_break_argmax
- [x] **44.** tic_tac_toe_reward
- [x] **45.** q_learning_nonterminal_target
- [x] **46.** q_learning_terminal_target
- [x] **47.** q_learning_update
- [x] **48.** episode_reset_game
- [x] **49.** episode_agent_pick_action
- [x] **50.** episode_apply_action
- [x] **51.** episode_apply_q_update
- [x] **52.** episode_check_terminate
- [x] **53.** train_q_learning_agent
- [x] **54.** compute_batched_outcome_stats
- [x] **55.** self_play_episode
- [x] **56.** flip_board_perspective
- [x] **57.** perspective_reward_sign
- [x] **58.** train_q_agent_self_play
- [x] **59.** evaluate_q_agent_vs_random
- [x] **60.** evaluate_q_agent_vs_minimax
- [x] **61.** inspect_q_values_for_state
- [x] **62.** serialize_q_table_to_dict
- [x] **63.** deserialize_q_table_from_dict
- [x] **64.** encode_board_flat_length_nine
- [x] **65.** encode_board_one_hot_length_eighteen
- [x] **66.** build_mlp_architecture
- [x] **67.** initialize_mlp_parameters
- [x] **68.** mlp_forward_pass
- [x] **69.** mask_illegal_actions_neg_inf
- [x] **70.** argmax_action_from_q_values
- [x] **71.** mse_loss_on_chosen_action
- [x] **72.** mlp_backward_pass
- [x] **73.** adam_update_step
- [x] **74.** create_replay_buffer
- [x] **75.** append_transition_to_buffer
- [x] **76.** cap_buffer_size_drop_oldest
- [x] **77.** sample_minibatch_from_buffer
- [x] **78.** build_target_network_copy
- [x] **79.** compute_target_q_with_target_network
- [x] **80.** sync_target_network_periodically
- [x] **81.** dqn_select_action
- [x] **82.** dqn_train_step
- [x] **83.** train_dqn_agent
- [x] **84.** compare_dqn_tabular_random_minimax
- [x] **85.** sarsa_on_policy_update
- [x] **86.** train_sarsa_agent
- [x] **87.** reinforce_log_prob_of_action
- [x] **88.** reinforce_collect_episode_returns
- [x] **89.** reinforce_policy_gradient_update
- [ ] **90.** train_reinforce_agent
- [ ] **91.** compare_value_vs_policy_learners
- [ ] **92.** symmetry_augmented_training

---

Built on Deep-ML.
