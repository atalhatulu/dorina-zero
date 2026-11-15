"""Kusursuz Connect 4 Çözücüsü (Pascal Pons GameSolver API İstemcisi)."""
import time
import requests

SOLVER_URL = "https://connect4.gamesolver.org/solve"
_CACHE = {}


def get_solver_best_move(move_history: list):
    """
    move_history: [3, 3, 2, 4, ...] gibi 0-indeksli hamle listesi.
    Dönüş: (en_iyi_hamle, hamle_skorlari, durum_metni)
    
    Pascal Pons Skor Anlamı:
      > 0 : Oynayan taraf kesin kazanıyor (Skor = 22 - (kalan hamle/2))
      = 0 : Kesin beraberlik
      < 0 : Oynayan taraf kaybediyor (Skor = -22 + (kalan hamle/2))
    """
    # 1-indeksli string formatına dönüştür (örn: "4435")
    pos_str = "".join(str(m + 1) for m in move_history)

    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    if pos_str in _CACHE:
        scores = _CACHE[pos_str]
    else:
        for attempt in range(3):
            try:
                resp = requests.get(f"{SOLVER_URL}?pos={pos_str}", headers=headers, timeout=6)
                if resp.status_code == 200:
                    data = resp.json()
                    scores = data.get("score", [])
                    _CACHE[pos_str] = scores
                    break
            except Exception:
                time.sleep(0.5)
        else:
            # API yanıt vermezse None dön
            return None, None, "API Bağlantı Hatası"

    # En yüksek skora sahip yasal hamleyi seç
    best_col = None
    best_score = -9999
    for col, sc in enumerate(scores):
        if sc == 100 or sc == -1000:  # Geçersiz/Dolu sütun
            continue
        if sc > best_score:
            best_score = sc
            best_col = col

    status = "Beraberlik" if best_score == 0 else ("Kazanıyor" if best_score > 0 else "Kaybediyor")
    return best_col, scores, f"{status} (Skor: {best_score})"


if __name__ == "__main__":
    col, sc, stat = get_solver_best_move([3, 3, 3])
    print(f"Test Başarılı: Seçilen Hamle={col}, Skorlar={sc}, Durum={stat}")
