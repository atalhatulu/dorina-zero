"""Model karşılaştırma: yeni vs eski, yeni vs minimax, Elo hesabı."""
import numpy as np
import torch

from board import (
    create_board, drop_piece, get_next_open_row,
    get_valid_locations, winning_move, is_full,
    PLAYER, AI,
)
from network import Connect4Net
from mcts import MCTS
import logger


def play_match(model_a, model_b, num_games=20, num_simulations=50,
               model_a_player=PLAYER, verbose=False):
    """
    İki modeli karşı karşıya oynat.
    model_a_player: A'nın hangi oyuncu olduğu (PLAYER veya AI)
    Dönüş: (a_wins, b_wins, draws)
    """
    model_a.eval()
    model_b.eval()
    mcts_a = MCTS(model_a, num_simulations=num_simulations)
    mcts_b = MCTS(model_b, num_simulations=num_simulations)

    a_wins, b_wins, draws = 0, 0, 0

    for i in range(num_games):
        board = create_board()
        current_player = PLAYER
        # Her oyunda A/B tarafını değiştir (adil)
        a_is_player = (model_a_player == PLAYER) if i % 2 == 0 else (model_a_player != PLAYER)

        while True:
            valid = get_valid_locations(board)
            if not valid:
                draws += 1
                break

            is_a_turn = (current_player == PLAYER) == a_is_player
            mcts = mcts_a if is_a_turn else mcts_b

            policy = mcts.search(board, current_player, temperature=0)
            move = int(np.argmax(policy))

            row = get_next_open_row(board, move)
            drop_piece(board, row, move, current_player)

            if winning_move(board, current_player):
                if is_a_turn:
                    a_wins += 1
                else:
                    b_wins += 1
                break
            if is_full(board):
                draws += 1
                break
            current_player = AI if current_player == PLAYER else PLAYER

        if verbose and (i + 1) % 5 == 0:
            print(f"  [{i+1}/{num_games}] A={a_wins} B={b_wins} D={draws}")

    return a_wins, b_wins, draws


def evaluate_vs_minimax(model, num_games=20, num_simulations=100,
                        minimax_depth=4, verbose=False):
    """Model vs minimax AI. (wins, losses, draws) döndür."""
    from minimax_ai import get_ai_move

    model.eval()
    mcts = MCTS(model, num_simulations=num_simulations)
    wins, losses, draws = 0, 0, 0

    for i in range(num_games):
        board = create_board()
        current_player = PLAYER
        model_player = PLAYER if i % 2 == 0 else AI

        while True:
            valid = get_valid_locations(board)
            if not valid:
                draws += 1
                break

            if current_player == model_player:
                policy = mcts.search(board, current_player, temperature=0)
                move = int(np.argmax(policy))
            else:
                move = get_ai_move(board, depth=minimax_depth)

            row = get_next_open_row(board, move)
            drop_piece(board, row, move, current_player)

            if winning_move(board, current_player):
                if current_player == model_player:
                    wins += 1
                else:
                    losses += 1
                break
            if is_full(board):
                draws += 1
                break
            current_player = AI if current_player == PLAYER else PLAYER

        if verbose and (i + 1) % 5 == 0:
            print(f"  [{i+1}/{num_games}] W={wins} L={losses} D={draws}")

    return wins, losses, draws


def compare_checkpoints(path_a, path_b, num_games=20, num_simulations=50,
                        iteration=None, verbose=True):
    """
    İki checkpoint dosyasını karşılaştır, eval.csv'ye logla.
    Dönüş: (a_wins, b_wins, draws, elo_diff)
    """
    model_a = Connect4Net()
    model_b = Connect4Net()
    from train import load_checkpoint
    it_a = load_checkpoint(model_a, path_a)
    it_b = load_checkpoint(model_b, path_b)

    if verbose:
        print(f"  Model A: iter {it_a} ({path_a})")
        print(f"  Model B: iter {it_b} ({path_b})")

    a_wins, b_wins, draws = play_match(
        model_a, model_b, num_games=num_games,
        num_simulations=num_simulations, verbose=verbose,
    )

    total = a_wins + b_wins + draws
    win_rate = a_wins / total if total else 0.0
    elo = logger.elo_diff(win_rate)

    if verbose:
        print(f"  Sonuç: A={a_wins} B={b_wins} D={draws} "
              f"(A kazanma: {win_rate:.1%}, Elo: {elo:+.0f})")

    if iteration is not None:
        logger.log_eval(iteration, it_b, a_wins, b_wins, draws, elo)

    return a_wins, b_wins, draws, elo