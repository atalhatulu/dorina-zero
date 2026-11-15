# 🟢 dorina4 — AlphaZero Othello (Reversi) Geçiş Raporu ve Mimari Doğrulama

> **Tarih:** 15 Kasım 2026  
> **Konu:** Oyun-Agnostik AlphaZero Motorunun Othello (8x8) için Devreye Alınması  
> **Durum:** Başarılı ve Doğrulandı (Tüm Testler ve İlk Canlı İterasyon Geçti)

---

## 1. Mimari Kararlar ve Yapılan Değişiklikler

### 1.1. Ağ Mimarisi: `network_othello.py` (`OthelloNet`)
* **Girdi Şekli:** `(batch, 3, 8, 8)`
  * Kanal 0: Sıradaki oyuncunun taşları (1.0 / 0.0)
  * Kanal 1: Rakip oyuncunun taşları (1.0 / 0.0)
  * Kanal 2: 1'lerden oluşan sabit kanal (renk/sıra kanalı)
* **Omurga:** 4 Residual Blok (64 kanal, 3x3 Conv, BatchNorm, ReLU, Skip connection)
* **Çift Başlık (Two-Headed):**
  * **Policy Head:** 1x1 Conv (4 kanal) + BN + Flatten + Linear -> **65 logits** (64 tahta karesi + 1 pass hamlesi)
  * **Value Head:** 1x1 Conv (2 kanal) + BN + Flatten + Linear(64) + Linear(1) -> **tanh [-1, +1]**
* **Parametre Sayısı:** 323,214 parametre (CPU üzerinde hızlı paralel çıkarım ve yüksek taktiksel temsil gücü).

### 1.2. 8x Dihedral Simetri Veri Artırımı (D4)
* Othello tahtası kare (8x8) olduğundan 4 rotasyon ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) ve bunların her birinin yatay yansıması (flip) tahta dinamiklerini tam korur.
* **Pass Hamlesi Koruması:** `augment_data()` fonksiyonunda ilk 64 kare 8x8 matrise çevrilip $D_4$ grubu simetrisi uygulanırken, 64. indeksteki **Pass hamlesi sabit bırakılarak** vektörün sonuna eklenir.
* **Sonuç:** Oynanan her maçtan **8 katı kadar zengin eğitim verisi** elde edilir.

### 1.3. Oyun Motoru Uyumluluğu: `games/othello.py`
* `_board_shape` özelliği `(3, 8, 8)` olarak güncellendi.
* `encode_state()` metodu Connect 4 standardıyla birebir uyumlu olarak 3. kanala $1.0$ sabit matrisini ekleyecek şekilde düzenlendi.

### 1.4. Dinamik Eğitim Pipeline'ı: `train_generic.py`
* Hardcoded `Connect4NetV2` ve `Connect4Game` bağımlılıkları kaldırılarak `get_game_and_model_factories(args.game)` yapısına geçildi.
* Hem self-play aşamasında hem de historical gating maçlarında oyun tipine göre uygun model ve oyun motoru eşlendi.

---

## 2. Doğrulama ve Test Sonuçları

1. **Birim Testler (`tests/test_othello.py`):**
   * Othello kuralları, geçerli hamleler (başlangıçta 19, 26, 37, 44) ve taş çevirme mantığı test edildi.
   * `OthelloNet` forward pass çıktı şekilleri `(B, 65)` ve `(B, 1)` doğrulandı.
   * $D_4$ simetrisinde pass değerinin korunduğu ve 8 katı tensör üretildiği teyit edildi.
   * MCTS tek adım aramasında yalnızca geçerli hamlelere olasılık atandığı doğrulandı.
   * **Sonuç:** `tests/` klasöründeki tüm testler (**70 test**) sıfır hatayla geçti.

2. **İlk Canlı Test İterasyonu:**
   * Komut: `python train_generic.py --game othello --iterations 1 --games-per-iter 4 --sims 20 --workers 2`
   * 4 self-play oyunu oynandı, 8x simetri ile 1920 eğitim tensörü oluşturuldu.
   * Sinir ağı 1 epoch eğitildi (`policy_loss=4.1591`, `value_loss=0.6953`).
   * Gating maçı başarıyla yapıldı.

---

## 3. Eğitimi Başlatma Komutu

Othello için sıfırdan çoklu işlemcili tam eğitimi başlatmak için:

```bash
./venv/bin/python train_generic.py \
  --game othello \
  --iterations 30 \
  --games-per-iter 100 \
  --sims 150 \
  --epochs 3 \
  --batch-size 64 \
  --lr 2e-4 \
  --workers 11 \
  --gating-every 1 \
  --gating-games 30 \
  --gating-sims 150 \
  --gating-threshold 0.55 \
  --out-dir checkpoints_othello
```
