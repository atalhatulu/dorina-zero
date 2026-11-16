# Othello AlphaZero vs Heuristic Minimax - Faz 4 Benchmark Raporu (İterasyon 47)

**Tarih:** 16 Kasım 2026  
**Eğitim Paketi:** Büyük Taarruz Paketi (İterasyon 29 - 48, 200 Sim, LR=5e-5, Minimax Depth 4 Curriculum)  
**Aktif Şampiyon:** `models/othello/checkpoints/current_best.pt` (İterasyon 47)  
**Rakip:** Othello Heuristic Minimax Bot (Derinlik 4, Konumsal Ağırlık Matrisi + Mobilite Boğması + Taş Paritesi)  

---

## 1. Yönetici Özeti: Minimax Mağlubiyetsiz Geçildi

- **Tarihi Kırılma:** AlphaZero, Minimax Derinlik 4 karşısında çıktığı 4 maçın **hiçbirini kaybetmedi (%0.0 Minimax Galibiyeti)**!
- **Nihai Skor Oranı:** **%75.0 Puan Oranı** (2 Galibiyet, 2 Beraberlik, 0 Mağlubiyet).
- **Evrim Basamakları:**
  - **İterasyon 14:** Minimax karşısında 0/4 (%0.0 puan) — X-karelerine basarak dağılıyordu.
  - **İterasyon 28:** Minimax karşısında 2/4 (%50.0 puan) — Köşe savunmasını keşfetti, maçı dengeledi.
  - **İterasyon 47 (Şu Anki Model):** Minimax karşısında **2 Galibiyet, 2 Beraberlik (%75.0 puan)** — Minimax'e tek bir galibiyet dahi vermedi.

---

## 2. Resmi Karşılaşma Sonuçları (MCTS 200 Simülasyon)

| Maç | AlphaZero Rolü | Minimax Rolü | Skor (AlphaZero - Minimax) | Kazanan | Hamle |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Siyah (P1) | Beyaz (P2) | 32 - 32 | **BERABERLİK** | 60 |
| 2 | **Beyaz (P2)** | Siyah (P1) | **34 - 30** | **ALPHAZERO** | 61 |
| 3 | Siyah (P1) | Beyaz (P2) | 32 - 32 | **BERABERLİK** | 60 |
| 4 | **Beyaz (P2)** | Siyah (P1) | **34 - 30** | **ALPHAZERO** | 61 |

---

## 3. Taktiksel Gelişim Analizi

1. **Siyah (P1) İken Denge ve Beraberlik (32 - 32):**
   - Siyah başladığında daha önce 13-50 gibi ağır kayıplar veren açılış zafiyeti tamamen kapatıldı.
   - Model, Minimax'in tüm tuzaklarını savuşturarak tahtayı milimetrik bir 32-32 eşitlikle kilitledi.
2. **Beyaz (P2) İken Tam Hakimiyet (34 - 30):**
   - Beyaz oynarken temposunu koruyup son periyotta 34-30 üstünlük kurarak Minimax'i devirdi.
3. **Konumsal Disiplin:**
   - 14. iterasyonda görülen "intihar karesi (X-square)" basma zaafı ortadan kalktı. Model artık köşeyi vermemek için kenar kilitlerini tercih ediyor.

---

## 4. Sıradaki Hamle: Usta Merdiveninde 2. Basamak (Derinlik 6)

Connect4 yol haritamızda olduğu gibi:
1. **Basamak 1 (Minimax Derinlik 4):** **AŞILDI** (Minimax 0 galibiyet, AlphaZero %75 puan).
2. **Basamak 2 (Minimax Derinlik 6):** 6 hamlelik derin taktiksel arama yapan versiyona karşı ringe çıkma zamanı.
3. **Basamak 3 (Turnuva Şampiyonu / Edax):** Final sınavı.
