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
- [ ] **22.** compute_outcome_rates
- [ ] **23.** minimax_terminal_score
- [ ] **24.** minimax_value
- [ ] **25.** minimax_recursive
- [ ] **26.** minimax_max_min_step
- [ ] **27.** minimax_best_move
- [ ] **28.** minimax_alpha_beta
- [ ] **29.** play_minimax_vs_random_matches
- [ ] **30.** play_minimax_vs_minimax_matches
- [ ] **31.** encode_board_state_key
- [ ] **32.** canonical_board_key
- [ ] **33.** initialize_q_table
- [ ] **34.** get_q_value
- [ ] **35.** set_q_value
- [ ] **36.** choose_learning_rate_alpha
- [ ] **37.** choose_discount_factor_gamma
- [ ] **38.** choose_initial_epsilon
- [ ] **39.** epsilon_decay_schedule
- [ ] **40.** epsilon_greedy_explore_move
- [ ] **41.** epsilon_greedy_select_action
- [ ] **42.** greedy_argmax_over_legal_actions
- [ ] **43.** random_tie_break_argmax
- [ ] **44.** tic_tac_toe_reward
- [ ] **45.** q_learning_nonterminal_target
- [ ] **46.** q_learning_terminal_target
- [ ] **47.** q_learning_update
- [ ] **48.** episode_reset_game
- [ ] **49.** episode_agent_pick_action
- [ ] **50.** episode_apply_action
- [ ] **51.** episode_apply_q_update
- [ ] **52.** episode_check_terminate
- [ ] **53.** train_q_learning_agent
- [ ] **54.** compute_batched_outcome_stats
- [ ] **55.** self_play_episode
- [ ] **56.** flip_board_perspective
- [ ] **57.** perspective_reward_sign
- [ ] **58.** train_q_agent_self_play
- [ ] **59.** evaluate_q_agent_vs_random
- [ ] **60.** evaluate_q_agent_vs_minimax
- [ ] **61.** inspect_q_values_for_state
- [ ] **62.** serialize_q_table_to_dict
- [ ] **63.** deserialize_q_table_from_dict
- [ ] **64.** encode_board_flat_length_nine
- [ ] **65.** encode_board_one_hot_length_eighteen
- [ ] **66.** build_mlp_architecture
- [ ] **67.** initialize_mlp_parameters
- [ ] **68.** mlp_forward_pass
- [ ] **69.** mask_illegal_actions_neg_inf
- [ ] **70.** argmax_action_from_q_values
- [ ] **71.** mse_loss_on_chosen_action
- [ ] **72.** mlp_backward_pass
- [ ] **73.** adam_update_step
- [ ] **74.** create_replay_buffer
- [ ] **75.** append_transition_to_buffer
- [ ] **76.** cap_buffer_size_drop_oldest
- [ ] **77.** sample_minibatch_from_buffer
- [ ] **78.** build_target_network_copy
- [ ] **79.** compute_target_q_with_target_network
- [ ] **80.** sync_target_network_periodically
- [ ] **81.** dqn_select_action
- [ ] **82.** dqn_train_step
- [ ] **83.** train_dqn_agent
- [ ] **84.** compare_dqn_tabular_random_minimax
- [ ] **85.** sarsa_on_policy_update
- [ ] **86.** train_sarsa_agent
- [ ] **87.** reinforce_log_prob_of_action
- [ ] **88.** reinforce_collect_episode_returns
- [ ] **89.** reinforce_policy_gradient_update
- [ ] **90.** train_reinforce_agent
- [ ] **91.** compare_value_vs_policy_learners
- [ ] **92.** symmetry_augmented_training

---

Built on Deep-ML.
