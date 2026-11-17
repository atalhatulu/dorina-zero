# Othello AlphaZero Turnuva Benchmark Raporu (Faz 10 - İterasyon 125)

**Tarih:** 18 Kasım 2026  
**Model:** `models/othello/checkpoints/current_best.pt` (İterasyon 125 Şampiyonu)  
**Ağ:** OthelloNet (65 aksiyon, 3x8x8 bitboard girdi)  
**MCTS:** 200 Simülasyon  

---

## 1. Yönetici Özeti
İterasyon 125 ile birlikte modelin **Edax Seviye 1'e karşı hakimiyeti %100'e ulaşmıştır**. Önceki sürümlerdeki Siyah/Beyaz asimetrisi tamamen giderilmiş, model hem Siyah hem Beyaz taşlarla Edax Seviye 1'i 4 maçta 4 galibiyetle geçmiştir.

Ancak **Edax Seviye 2** seviyesine geçildiğinde model derin taktiksel arama karşısında yenilmiştir (0/4). Bu durum, modelin artık Seviye 1 ligini bitirip doğrudan **Edax Seviye 2 müfredatına** geçmesi gerektiğini kesinleştirmiştir.

---

## 2. Detaylı Karşılaşma Skorları

### A. Edax Tournament Engine (Seviye 1) — %100 Hakimiyet
- **Toplam Maç:** 4
- **AlphaZero Galibiyeti:** 4 / 4 (%100.0)
- **Edax Galibiyeti:** 0 / 4 (%0.0)
- **Ortalama Taş Sayısı:** Model 34.5 vs 29.5 Edax

| Maç | Model Rengi | Skor | Sonuç | Hamle Sayısı |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Siyah (P1) | 33 - 31 | **MODEL KAZANDI** | 60 |
| 2 | Beyaz (P2) | 36 - 28 | **MODEL KAZANDI** | 60 |
| 3 | Siyah (P1) | 33 - 31 | **MODEL KAZANDI** | 60 |
| 4 | Beyaz (P2) | 36 - 28 | **MODEL KAZANDI** | 60 |

---

### B. Edax Tournament Engine (Seviye 2) — Üst Lig Sınavı
- **Toplam Maç:** 4
- **AlphaZero Galibiyeti:** 0 / 4 (%0.0)
- **Edax Galibiyeti:** 4 / 4 (%100.0)
- **Ortalama Taş Sayısı:** Model 13.0 vs 51.0 Edax

| Maç | Model Rengi | Skor | Sonuç | Hamle Sayısı |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Siyah (P1) | 15 - 49 | EDAX KAZANDI | 61 |
| 2 | Beyaz (P2) | 11 - 53 | EDAX KAZANDI | 65 |
| 3 | Siyah (P1) | 15 - 49 | EDAX KAZANDI | 61 |
| 4 | Beyaz (P2) | 11 - 53 | EDAX KAZANDI | 65 |

---

### C. Heuristic Minimax (Derinlik 4) Karşılaşması
- **Toplam Maç:** 4
- **AlphaZero Galibiyeti:** 2 / 4 (%50.0)
- **Minimax Galibiyeti:** 2 / 4 (%50.0)
- **Önemli An:** Beyaz taşlarla Minimax'ı 16 hamlede 20-0 ile nakavt etmiştir.

---

## 3. Teşhis ve Sıradaki Adım

1. **Başarı:** Edax Seviye 1 seviyesi tam olarak çözüldü ve ekarte edildi.
2. **Kritik Eksik:** Edax Seviye 2'nin arama ağacı daha derin olduğu için açılışta ve orta oyun başında model erken taş kazanma tuzağına düşmekte, Edax ise köşeleri alarak 50+ taş farkıyla kazanmaktadır.
3. **Aksiyon:** Kaggle eğitiminde `--curriculum-depth 2` ile Edax Seviye 2 öğretmen olarak atanmalı, yumuşatılmış politika (Soft Teacher) ile Seviye 2 hamleleri ağa öğretilmelidir.
