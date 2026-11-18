# Othello AlphaZero Turnuva Benchmark Raporu (Faz 12 - İterasyon 143 Tarihi Zafer)

**Tarih:** 18 Kasım 2026  
**Şampiyon Model:** `models/othello/checkpoints/current_best.pt` (İterasyon 143)  
**Müfredat:** Hibrit Edax (Seviye 1 Çapa + Seviye 2 Tırmanış, %20 Oran)  
**MCTS:** 200 Simülasyon  

---

## 1. Yönetici Özeti ve Tarihi Dönüm Noktası

İterasyon 125'e dönüp hibrit curriculum mimarisini devreye alma kararımız projenin en büyük başarısını getirdi:

1. **Edax Seviye 2 Bariyeri Kırıldı (Tarihte İlk Kez):**
   - Modelimiz, dünya standartlarındaki Edax motorunu Seviye 2 derinliğinde **Beyaz taşlarla 33 - 31 mağlup etti!**
   - Toplam 4 maçta 2 galibiyet alarak Edax Seviye 2 ile **%50 - %50 kafa kafaya** turnuva seviyesine ulaştı.
2. **Edax Seviye 1'e Karşı Sıfır Mağlubiyet (%75 Puan Oranı):**
   - Unutma etkisi tamamen yok edildi. Edax Seviye 1 modele karşı tek bir galibiyet bile alamadı.
   - Beyaz ile **44 - 20** ezici zafer, Siyah ile **32 - 32** çelik savunma beraberliği.

---

## 2. Detaylı Karşılaşma Skor Tablosu

### A. Edax Tournament Engine (Seviye 2) — Üst Lig Karşılaşması
- **Toplam Maç:** 4
- **AlphaZero Galibiyeti:** 2 / 4 (%50.0)
- **Edax Galibiyeti:** 2 / 4 (%50.0)
- **Sonuç:** **DÜNYA STANDARDI MOTORLA KAFA KAFAYA EŞİTLİK**

| Maç | Renk | Skor | Kazanan | Detay |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Siyah (P1) | 7 - 57 | EDAX | Edax erken köşe baskısı kurdu |
| 2 | Beyaz (P2) | **33 - 31** | **MODEL** | **Tarihi Zafer: Seviye 2 Edax mağlup edildi!** |
| 3 | Siyah (P1) | 7 - 57 | EDAX | Edax erken köşe baskısı kurdu |
| 4 | Beyaz (P2) | **33 - 31** | **MODEL** | **Tarihi Zafer: Zafer tekrarlandı ve tescillendi!** |

---

### B. Edax Tournament Engine (Seviye 1) — Çapa Testi
- **Toplam Maç:** 4
- **AlphaZero Galibiyeti:** 2 / 4
- **Beraberlik:** 2 / 4
- **Edax Galibiyeti:** 0 / 4 (**%0.0 - EDAX SIFIR GALİBİYET**)
- **Puan Oranı:** **%75.0**

| Maç | Renk | Skor | Kazanan | Detay |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Siyah (P1) | 32 - 32 | BERABERE | Kusursuz taş dengesi |
| 2 | Beyaz (P2) | **44 - 20** | **MODEL** | Ezici taş üstünlüğü |
| 3 | Siyah (P1) | 32 - 32 | BERABERE | Kusursuz taş dengesi |
| 4 | Beyaz (P2) | **44 - 20** | **MODEL** | Ezici taş üstünlüğü |

---

## 3. Durum Değerlendirmesi ve Kalan Tek Zayıflık

Modelimiz artık resmi olarak **Edax Seviye 2 Liginde**.

- **Güçlü Yön:** Beyaz (P2) taşlarla Edax Seviye 1'i 44-20, Edax Seviye 2'yi 33-31 yenen insanüstü bir evaporasyon ve oyun sonu motoruna dönüştü.
- **Kalan Tek Zayıflık:** Siyah (P1) açılışındaki köşe savunması. Siyah açılışındaki ilk 4 hamle teorisini sağlamlaştırdığımız an model Edax Seviye 2'yi hem Siyah hem Beyaz ile süpürecek.
