# Othello AlphaZero Faz 8 Benchmark Raporu (İterasyon 91 - Soft Teacher)

**Tarih:** 17 Kasım 2026  
**Aktif Şampiyon:** `models/othello/checkpoints/current_best.pt` (İterasyon 91)  
**Yöntem:** MCTS Harmanlamalı Yumuşatılmış Öğretici (Soft Teacher: %30 MCTS + %70 Edax)  

---

## 1. Yönetici Özeti: Zehirlenme Giderildi, Stabilite Sağlandı

- **Önceki Kriz (İterasyon 94):** %100 sert Dirac etiketi yüzünden `value_loss` 0.30'a fırlamış, model hem Edax'a hem Minimax'e karşı %0'a düşmüştü.
- **Şimdiki Durum (İterasyon 91):**
  - `value_loss` 0.16 - 0.24 bandında stabil kaldı.
  - **Minimax Depth 4:** **4 / 4 (%100.0 Galibiyet)** — Siyah 36-28, Beyaz 34-30.
  - **Edax Level 1:** **2 / 4 (%50.0 Galibiyet)** — Siyah 35-29 galip, Beyaz 3-60 mağlup.

---

## 2. Karşılaştırmalı Test Sonuçları

| Rakip | Model Rolü | Skor (Model - Rakip) | Kazanan |
| :--- | :---: | :---: | :--- |
| **Minimax Depth 4** | Siyah (P1) | **36 - 28** | **AlphaZero** |
| **Minimax Depth 4** | Beyaz (P2) | **34 - 30** | **AlphaZero** |
| **Edax Level 1** | Siyah (P1) | **35 - 29** | **AlphaZero** |
| **Edax Level 1** | Beyaz (P2) | 3 - 60 | Edax |

---

## 3. Sıradaki Adım

Model Siyah (P1) ile Edax Level 1'i yenebiliyor ve Minimax'i %100 eziyor. Tek kalan açık **Beyaz (P2) ile Edax açılışına karşı savunma**.
Bir sonraki adımda:
- Mevcut İterasyon 91 şampiyonu üzerinden Edax Level 1'e karşı **Beyaz (P2) açılışlarını** pekiştirecek kısa bir Kaggle paketi veya doğrudan Edax Level 2 testi yapılabilir.
