import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import pytest

import logger
from network import Connect4Net
from self_play import play_one_game, history_to_training_data, play_games


@pytest.fixture(autouse=True)
def clean_logs():
    if os.path.exists(logger.LOGS_DIR):
        shutil.rmtree(logger.LOGS_DIR)
    yield
    if os.path.exists(logger.LOGS_DIR):
        shutil.rmtree(logger.LOGS_DIR)


@pytest.fixture
def model():
    return Connect4Net()


def test_play_one_game_returns_valid(model):
    history, winner, moves = play_one_game(
        model, num_simulations=10, log=False,
    )
    assert winner in ("player1", "player2", "draw")
    assert moves > 0
    assert len(history) == moves
    for board, player, policy in history:
        assert player in (1, 2)
        assert policy.shape == (7,)
        assert abs(policy.sum() - 1.0) < 1e-4


def test_history_to_training_data_player1_win(model):
    history, winner, _ = play_one_game(model, num_simulations=5, log=False)
    data = history_to_training_data(history, "player1")
    assert len(data) == len(history) * 2  # symmetry augmentation
    for board, player, policy, v in data:
        if player == 1:
            assert v == 1.0
        else:
            assert v == -1.0


def test_history_to_training_data_draw(model):
    history, _, _ = play_one_game(model, num_simulations=5, log=False)
    data = history_to_training_data(history, "draw")
    assert len(data) == len(history) * 2  # symmetry augmentation
    for _, _, _, v in data:
        assert v == 0.0


def test_play_games_stats(model):
    _, stats = play_games(model, num_games=3, num_simulations=5,
                          iteration=0, log=False)
    total = stats["player1"] + stats["player2"] + stats["draw"]
    assert total == 3
    assert stats["avg_moves"] > 0


def test_play_one_game_logs(model):
    play_one_game(model, num_simulations=5, game_id=1, iteration=0, log=True)
    assert os.path.exists(logger.GAMES_PATH)
    with open(logger.GAMES_PATH) as f:
        lines = f.readlines()
    assert len(lines) == 1
