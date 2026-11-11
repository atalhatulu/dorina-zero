"""Pozisyon coverage tracking. Benzersiz pozisyonları hash'le, istatistik tut."""
import hashlib
import json
import os

COVERAGE_DIR = "logs"
COVERAGE_FILE = os.path.join(COVERAGE_DIR, "coverage.json")
VISITED_FILE = os.path.join(COVERAGE_DIR, "visited_hashes.txt")

# Connect4 toplam legal pozisyon sayısı (Tromp, 1995)
TOTAL_POSITIONS = 4_531_985_219_092


def board_hash(board, player):
    """Tahta + sıra kimde → benzersiz hash."""
    # Board'u bytes'a çevir
    flat = bytes(cell for row in board for cell in row)
    # Player'ı ekle (sıra önemli)
    data = flat + bytes([player])
    return hashlib.md5(data).hexdigest()[:16]  # 16 char yeterli


class CoverageTracker:
    def __init__(self, visited_file=VISITED_FILE):
        self.visited_file = visited_file
        self.visited = set()
        self._load()
    
    def _load(self):
        os.makedirs(COVERAGE_DIR, exist_ok=True)  # ← EKLE
        if os.path.exists(self.visited_file):
            with open(self.visited_file) as f:
                self.visited = set(line.strip() for line in f if line.strip())
    
    def add(self, board, player):
        """Pozisyonu kaydet. Yeni ise True döndür."""
        h = board_hash(board, player)
        if h not in self.visited:
            self.visited.add(h)
            os.makedirs(COVERAGE_DIR, exist_ok=True)  # ← EKLE
            with open(self.visited_file, "a") as f:
                f.write(h + "\n")
            return True
        return False
    
    def stats(self):
        n = len(self.visited)
        return {
            "unique_positions": n,
            "total_positions": TOTAL_POSITIONS,
            "coverage_pct": (n / TOTAL_POSITIONS) * 100,
            "coverage_str": f"{n:,} / {TOTAL_POSITIONS:,}",
        }
    
    def save_stats(self):
        os.makedirs(COVERAGE_DIR, exist_ok=True)
        with open(COVERAGE_FILE, "w") as f:
            json.dump(self.stats(), f, indent=2)


# Global tracker (tek instance)
_tracker = None

def get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = CoverageTracker()
    return _tracker


def track_position(board, player):
    """Self-play'de her pozisyon için çağır."""
    return get_tracker().add(board, player)
