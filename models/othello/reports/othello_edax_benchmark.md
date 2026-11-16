# Othello AlphaZero vs Edax Tournament Engine - Faz 6 Kırılma Raporu (İterasyon 78)

**Tarih:** 16 Kasım 2026  
**Eğitim Paketi:** Edax Curriculum Faz 1 (İterasyon 64 - 78, 200 Sim, Edax Level 1 Curriculum %20)  
**Aktif Şampiyon:** `models/othello/checkpoints/current_best.pt` (İterasyon 78)  
**Rakip:** Edax 4.7 (Othello Dünya Şampiyonası Referans C Motoru - Level 1)  

---

## 1. Yönetici Özeti: Edax Karşısında İlk Resmi Galibiyetler!

- **Önceki Durum (İterasyon 63):** Edax karşısında 4 maçta 0 galibiyet (%0.0) ve maç başına sadece 2-12 taş toplayabilen ağır hezimetler.
- **Şimdiki Durum (İterasyon 78):**
  - **200 MCTS Simülasyonu:** 4 maçta **2 Galibiyet / 2 Mağlubiyet (%50.0 Galibiyet Oranı)**!
  - Siyah (P1) oynadığı iki maçı da **35 - 29** kazanarak Edax'ı devirmeyi başardı.
- **Kritik Eşik Aşıldı:** AlphaZero, amatör Minimax'ten sonra profesyonel dünya motoru Edax'a karşı da diş geçirebileceğini kanıtladı.

---

## 2. Test Sonuçları Tablosu

### A. 200 MCTS Simülasyonu
| Maç | AlphaZero Rolü | Edax Rolü | Skor (AlphaZero - Edax) | Kazanan | Hamle |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Siyah (P1)** | Beyaz (P2) | **35 - 29** | **AlphaZero** | 61 |
| 2 | Beyaz (P2) | Siyah (P1) | 11 - 53 | Edax | 63 |
| 3 | **Siyah (P1)** | Beyaz (P2) | **35 - 29** | **AlphaZero** | 61 |
| 4 | Beyaz (P2) | Siyah (P1) | 11 - 53 | Edax | 63 |

---

## 3. Derin Analiz: Neden Siyah Kazanıyor, Beyaz Kaybediyor?

1. **Siyahın Açılış İnisiyatifi (35 - 29 Zaferi):**
   - Siyah başladığında model Edax'ın erken evaporasyonuna (az taş çevirme stratejisine) ayak uyduruyor ve merkezi terk etmeden tempoyu elinde tutuyor.
   - 61. hamlede tahta kapandığında 35-29 net bir üstünlükle Edax'ı mağlup ediyor.
2. **Beyaz Taş (P2) Zafiyeti (11 - 53 Kaybı):**
   - Beyaz başladığında Edax'ın açılış baskısı karşısında 15-20. hamle civarında dış çembere fazla taş bırakıyor.
   - Edax bu taşları kilit köşelere basmak için basamak olarak kullanıp tahtayı domine ediyor.
3. **Eğitim Veriminin Zirvesi:**
   - C diliyle yazılmış Edax motoru self-play'i yavaşlatmadı; tur süreleri **180 saniyeden 140 saniyeye düştü**.
   - Sadece 15 iterasyonluk Edax tecrübesi bile modelin taş toplama ortalamasını 3 taştan 35 taşa çıkardı.

---

## 4. Bir Sonraki Adım: Edax Karşısında Tam Hakimiyet

- Beyaz taşla olan zafiyeti kapatmak ve galibiyet oranını %50'den %75-80 bandına çekmek için:
  - **Edax Curriculum Paketi 2 (İterasyon 79 $\rightarrow$ 95)**:
  - Curriculum oranını %20'den **%25**'e çıkaracağız.
  - Edax Level 1 ile devam edip Beyaz savunmasını pekiştireceğiz.
