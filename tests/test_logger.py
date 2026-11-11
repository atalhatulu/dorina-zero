import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import csv
import shutil
import pytest
import logger


@pytest.fixture(autouse=True)
def clean_logs():
    """Her testten önce logs/ temizle, sonra sil."""
    if os.path.exists(logger.LOGS_DIR):
        shutil.rmtree(logger.LOGS_DIR)
    yield
    if os.path.exists(logger.LOGS_DIR):
        shutil.rmtree(logger.LOGS_DIR)


def test_log_game_creates_file():
    logger.log_game(1, 1, 25, "player1", 1.23)
    assert os.path.exists(logger.GAMES_PATH)
    with open(logger.GAMES_PATH) as f:
        line = f.readline()
    rec = json.loads(line)
    assert rec["game_id"] == 1
    assert rec["winner"] == "player1"
    assert rec["moves"] == 25


def test_log_game_appends():
    logger.log_game(1, 1, 25, "player1", 1.0)
    logger.log_game(2, 1, 30, "player2", 1.5)
    with open(logger.GAMES_PATH) as f:
        lines = f.readlines()
    assert len(lines) == 2


def test_log_iteration_header():
    logger.log_iteration(1, 100, 52, 44, 4, 24.5, 1.1, 1.234, 0.456, 0.0)
    with open(logger.ITER_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["iteration"] == "1"
    assert rows[0]["p1_wins"] == "52"


def test_log_iteration_multiple_no_duplicate_header():
    logger.log_iteration(1, 100, 52, 44, 4, 24.5, 1.1)
    logger.log_iteration(2, 100, 60, 35, 5, 23.0, 1.0)
    with open(logger.ITER_PATH) as f:
        lines = f.readlines()
    assert len(lines) == 3  # header + 2 satır


def test_log_eval():
    logger.log_eval(2, 1, 58, 38, 4, 142.0)
    with open(logger.EVAL_PATH) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["wins"] == "58"
    assert rows[0]["win_rate"] == "0.58"


def test_elo_diff_symmetric():
    assert abs(logger.elo_diff(0.5)) < 0.01
    assert logger.elo_diff(0.75) > 100
    assert logger.elo_diff(0.25) < -100
