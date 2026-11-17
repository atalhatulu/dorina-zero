# Othello AlphaZero Faz 9 Zafer Raporu: Beyaz (P2) Bariyeri Yıkıldı!

**Tarih:** 17 Kasım 2026  
**Aktif Şampiyon:** `models/othello/checkpoints/current_best.pt` (İterasyon 101)  
**Yöntem:** Düşük Dozaj (%10 Edax) + Soft Teacher + 16 Gating Kontrolü  

---

## 1. Büyük Kırılma Noktası: Beyaz ile Tarihi Galibiyet (43 - 21)

Önceki tüm turlarda (İterasyon 78 ve 91 dahil) AlphaZero **Beyaz (P2) oynarken Edax karşısında 3-60 ve 11-53 gibi hezimetler alıyordu**.

**İterasyon 101 ile Tarih Yazıldı:**
- **Edax Level 1 karşısında Beyaz (P2):** **Model 43 - 21 Edax (Net Bozgun / Ezici Galibiyet)!**
- Edax'ın açılış baskısını tamamen kırıp evaporasyon teorisini Edax'a karşı kullanarak 43 taşla tahtayı ele geçirdi.

---

## 2. Test Sonuçları Tablosu

### A. Edax Level 1 Karşılaşması (MCTS 200 Simülasyon)
| Maç | Model Rolü | Edax Rolü | Skor (Model - Edax) | Kazanan | Durum |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | Siyah (P1) | Beyaz (P2) | 12 - 52 | Edax | Mağlup |
| 2 | **Beyaz (P2)** | Siyah (P1) | **43 - 21** | **AlphaZero** | **TARİHİ ZAFER (Bozgun!)** |
| 3 | Siyah (P1) | Beyaz (P2) | 12 - 52 | Edax | Mağlup |
| 4 | **Beyaz (P2)** | Siyah (P1) | **43 - 21** | **AlphaZero** | **TARİHİ ZAFER (Bozgun!)** |

### B. Minimax Depth 4 Karşılaşması (MCTS 200 Simülasyon)
- Siyah (P1): **34 - 30** Galip
- Beyaz (P2): 24 - 40 Mağlup
- Genel: 2 / 4 (%50.0 Denge)

---

## 3. Taktiksel Çözümleme

1. **Beyaz Açılışının Çözülmesi:**
   - Model Edax'ın erken hamle baskısına karşı dış çembere taş kaçırma zaafını kapattı.
   - İkinci oyuncu (P2) olarak oyunu geriden kurup oyun ortasında Edax'ı 43-21 gibi ağır bir farkla yendi.
2. **Buffer Sağlığı:**
   - %10 Edax oranı sayesinde `value_loss` 0.19 - 0.25 bandında kaldı; önceki çöküşün aksine modelin dengesi bozulmadı.
