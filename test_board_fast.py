"""tests/test_board.py testlerini board_fast üzerinde koşan doğrulama ve kıyaslama scripti."""
import time
import pytest
import board as original_board
import board_fast as fast_board


def run_compat_tests():
    """test_board.py test senaryolarını board_fast üzerinde çalıştır."""
    b = fast_board.create_board()
    assert len(b) == 6 and len(b[0]) == 7

    fast_board.drop_piece(b, 5, 0, fast_board.PLAYER)
    assert b[5][0] == fast_board.PLAYER

    clone = fast_board.clone_board(b)
    clone[5][0] = fast_board.AI
    assert b[5][0] == fast_board.PLAYER

    # Yatay galibiyet
    b_h = fast_board.create_board()
    for c in range(4):
        fast_board.drop_piece(b_h, 5, c, fast_board.PLAYER)
    assert fast_board.winning_move(b_h, fast_board.PLAYER)
    assert not fast_board.winning_move(b_h, fast_board.AI)

    # Dikey galibiyet
    b_v = fast_board.create_board()
    for r in range(2, 6):
        fast_board.drop_piece(b_v, r, 0, fast_board.AI)
    assert fast_board.winning_move(b_v, fast_board.AI)
    assert not fast_board.winning_move(b_v, fast_board.PLAYER)

    # Çapraz galibiyet (down-right)
    b_d1 = fast_board.create_board()
    for i in range(4):
        fast_board.drop_piece(b_d1, 2 + i, i, fast_board.PLAYER)
    assert fast_board.winning_move(b_d1, fast_board.PLAYER)

    # Çapraz galibiyet (up-right)
    b_d2 = fast_board.create_board()
    for i in range(4):
        fast_board.drop_piece(b_d2, 5 - i, i, fast_board.PLAYER)
    assert fast_board.winning_move(b_d2, fast_board.PLAYER)

    # 3 taş galibiyet sayılmamalı
    b_3 = fast_board.create_board()
    for c in range(3):
        fast_board.drop_piece(b_3, 5, c, fast_board.PLAYER)
    assert not fast_board.winning_move(b_3, fast_board.PLAYER)

    print("board_fast: Tüm uyumluluk testleri BAŞARIYLA geçti! ✅")


def benchmark_speed():
    b = original_board.create_board()
    for c in range(3):
        original_board.drop_piece(b, 5, c, original_board.PLAYER)
    original_board.drop_piece(b, 5, 3, original_board.PLAYER)  # 4'lü hazır

    N = 20_000

    # 1. Orijinal board.py
    t0 = time.time()
    for _ in range(N):
        original_board.winning_move(b, original_board.PLAYER)
    t_orig = time.time() - t0

    # 2. Bitboard (Saf bit düzeyinde kazanma fonksiyonu)
    p1, p2, _ = fast_board.board_to_bitboards(b)
    t0 = time.time()
    for _ in range(N):
        fast_board.bitboard_is_win(p1)
    t_fast_pure = time.time() - t0

    # 3. Bitboard (Köprü fonksiyon ile liste -> bitboard -> kontrol)
    t0 = time.time()
    for _ in range(N):
        fast_board.winning_move(b, fast_board.PLAYER)
    t_fast_bridge = time.time() - t0

    print(f"\n{N:,} Kez Kazanma Kontrolü (winning_move) Kıyaslaması:")
    print(f"  1. Orijinal board.py (Döngüler):            {t_orig*1000:.2f} ms")
    print(f"  2. board_fast.py (Köprü / Liste Uyumluluğu): {t_fast_bridge*1000:.2f} ms")
    print(f"  3. Saf Bitboard bitboard_is_win (Bitwise):    {t_fast_pure*1000:.2f} ms")
    print(f"\nSaf Bitboard Hızlanması: {t_orig / t_fast_pure:.1f}x KAT DAHA HIZLI! 🚀")


if __name__ == "__main__":
    run_compat_tests()
    benchmark_speed()
