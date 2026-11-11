"""board_fast.py — Pascal Pons Bitboard mimarisi (Connect4).
Mevcut board.py API uyumluluğunu korur (liste tabanlı girdi/çıktı köprüsüyle).
"""

ROWS, COLS = 6, 7
EMPTY, PLAYER, AI = 0, 1, 2


class FastBitboard:
    """Pascal Pons tarzı 7x7 bitboard (49 bit).

    Her sütun 7 bit yer kaplar (satır 0..5 oyun alanı, satır 6 sınır/taşma biti).
    col c'nin taban biti = c * 7.
    """
    __slots__ = ("mask", "current_player", "moves")

    def __init__(self, mask=0, current_player=0, moves=0):
        self.mask = mask                      # Dolu tüm kareler
        self.current_player = current_player  # Sıradaki oyuncunun taşları
        self.moves = moves                    # Hamle sayısı

    def can_play(self, col):
        # Sütunun en üst oynanabilir biti (satır 5, yani taban + 5)
        top_mask = 1 << (col * 7 + 5)
        return (self.mask & top_mask) == 0

    def play(self, col):
        """Hamle yap: mask'e bit ekle, oyuncu taşlarını tersine çevir."""
        # Pascal Pons sihirli eklemesi: (mask + taban_biti) o sütundaki ilk boş biti bulur
        move_bit = (self.mask + (1 << (col * 7))) & ((1 << (col * 7 + 6)) - 1)
        self.current_player ^= self.mask
        self.mask |= move_bit
        self.moves += 1

    def is_win(self, bitboard):
        """4'lü dizi kontrolü: yatay, dikey, iki çapraz (sadece 4 bit kaydırma ve AND)."""
        # Yatay (adım: 7)
        m = bitboard & (bitboard >> 7)
        if m & (m >> 14):
            return True

        # Çapraz \ (adım: 6)
        m = bitboard & (bitboard >> 6)
        if m & (m >> 12):
            return True

        # Çapraz / (adım: 8)
        m = bitboard & (bitboard >> 8)
        if m & (m >> 16):
            return True

        # Dikey (adım: 1)
        m = bitboard & (bitboard >> 1)
        if m & (m >> 2):
            return True

        return False


def board_to_bitboards(board):
    """Mevcut 2B liste tahtasını (p1_bits, p2_bits, mask) üçlüsüne dönüştürür."""
    p1 = 0
    p2 = 0
    mask = 0
    for c in range(COLS):
        col_base = c * 7
        for r in range(ROWS - 1, -1, -1):
            cell = board[r][c]
            bit_idx = col_base + (ROWS - 1 - r)
            if cell != EMPTY:
                bit = 1 << bit_idx
                mask |= bit
                if cell == PLAYER:
                    p1 |= bit
                elif cell == AI:
                    p2 |= bit
    return p1, p2, mask


def bitboard_is_win(bitboard):
    """Verilen oyuncu bitmask'inde 4'lü var mı kontrol eder (O(1))."""
    # Yatay (7)
    m = bitboard & (bitboard >> 7)
    if m & (m >> 14):
        return True
    # Çapraz \ (6)
    m = bitboard & (bitboard >> 6)
    if m & (m >> 12):
        return True
    # Çapraz / (8)
    m = bitboard & (bitboard >> 8)
    if m & (m >> 16):
        return True
    # Dikey (1)
    m = bitboard & (bitboard >> 1)
    if m & (m >> 2):
        return True
    return False


# --- board.py API Birebir Eşdeğerleri ---

def create_board():
    return [[EMPTY] * COLS for _ in range(ROWS)]


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[0][col] == EMPTY


def get_next_open_row(board, col):
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == EMPTY:
            return r
    return None


def get_valid_locations(board):
    return [c for c in range(COLS) if is_valid_location(board, c)]


def winning_move(board, piece):
    """Bitboard ile 4'lü kontrolü (For döngüsüz, ultra hızlı)."""
    p1, p2, _ = board_to_bitboards(board)
    target = p1 if piece == PLAYER else p2
    return bitboard_is_win(target)


def is_full(board):
    return all(board[0][c] != EMPTY for c in range(COLS))


def is_terminal(board):
    p1, p2, mask = board_to_bitboards(board)
    if bitboard_is_win(p1) or bitboard_is_win(p2):
        return True
    # 42 taş dolu mu? (Her sütunun en üst oynanabilir biti 1 mi?)
    for c in range(COLS):
        if (mask & (1 << (c * 7 + 5))) == 0:
            return False
    return True


def clone_board(board):
    return [row[:] for row in board]
