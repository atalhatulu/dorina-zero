"""tests/test_flip.py — Yatay Çevirme (Horizontal Flip) Doğrulama Testi.

1. Policy simetrisi: i ↔ 6 - i (Connect4 7 sütun: 0↔6, 1↔5, 2↔4, 3↔3)
2. State simetrisi: Tahtanın sütunlarının ters çevrilmesi
3. Value simetrisi: Yatay çevirme kazananı veya tahta değerini DEĞİŞTİRMEZ.
"""
import numpy as np
import pytest
from core.self_play import augment_with_horizontal_flip
from games.connect4 import Connect4Game


def test_flip_policy_index():
    # 7 sütunlu Connect4 için policy mapping
    original_policy = np.array([0.1, 0.2, 0.05, 0.3, 0.15, 0.1, 0.1], dtype=np.float32)
    flipped = original_policy[::-1].copy()

    for i in range(7):
        assert flipped[i] == original_policy[6 - i], f"Hata: {i} ile {6-i} eşleşmedi!"


def test_flip_value_and_state_consistency():
    game = Connect4Game()
    state = game.get_initial_state()
    # Hamleler: P1 sütun 1'e, P2 sütun 5'e oynasın
    state = game.get_next_state(state, 1, 1)
    state = game.get_next_state(state, 5, 2)

    encoded = game.encode_state(state, 1)
    policy = np.array([0.0, 0.8, 0.0, 0.1, 0.0, 0.1, 0.0], dtype=np.float32)

    history = [(encoded, policy, 1)]
    # P1 kazandı varsayalım
    aug_data = augment_with_horizontal_flip(history, win_player=1)

    assert len(aug_data) == 2, "1 geçmiş örneği 2 veri üretmeli (orijinal + flip)"

    orig_x, orig_p, orig_v = aug_data[0]
    flip_x, flip_p, flip_v = aug_data[1]

    # Value flip'ten etkilenmemeli
    assert orig_v == flip_v == 1.0, f"Value değişti: orig={orig_v}, flip={flip_v}"

    # Policy 1 (indeks 1) -> 6-1=5 olmalı
    assert orig_p[1] == 0.8
    assert flip_p[5] == 0.8
    assert flip_p[1] == orig_p[5] == 0.1

    # State encoding sütunları ters çevrilmiş olmalı
    # axis 2 sütun eksenidir (shape: (3, 6, 7))
    for c in range(7):
        np.testing.assert_array_equal(flip_x[:, :, c], orig_x[:, :, 6 - c])
