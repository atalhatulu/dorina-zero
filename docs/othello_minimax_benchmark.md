# Othello AlphaZero vs Heuristic Minimax - Faz 5 Şampiyonluk Raporu (İterasyon 63)

**Tarih:** 16 Kasım 2026  
**Eğitim Paketi:** Büyük Taarruz Faz 2 (İterasyon 48 - 67, 200 Sim, LR=5e-5, Minimax Curriculum)  
**Aktif Şampiyon:** `models/othello/checkpoints/current_best.pt` (İterasyon 63)  
**Rakipler:** Heuristic Minimax (Derinlik 4 ve Derinlik 5)  

---

## 1. Yönetici Özeti: Minimax Tamamen Ezildi (%100 Dominasyon)

- **Minimax Derinlik 4:** **4 Maçta 4 Galibiyet (%100.0 Galibiyet Oranı)**  
  - Siyah (P1): 37 - 27 Galip  
  - Beyaz (P2): 35 - 29 Galip  
- **Minimax Derinlik 5 (Daha Sert Usta):** **2 Maçta 2 Galibiyet (%100.0 Galibiyet Oranı)**  
  - Siyah (P1): **41 - 23 Net Bozgun**  
  - Beyaz (P2): **33 - 31 Galip**  
- **Özet:** Model hem Siyah hem Beyaz ile Minimax'e karşı tek bir maç veya beraberlik dahi vermeden tahtayı süpürdü.

---

## 2. Gelişim Yolculuğu (Tabula Rasa'dan Zirveye)

| Aşama / İterasyon | Minimax Derinliği | AlphaZero Galibiyet | Beraberlik | Minimax Galibiyet | Puan Oranı | Durum |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **İterasyon 14** | Depth 3 | 0 / 4 | 0 | 4 / 4 | **%0.0** | Dağılma (X-karelerine intihar hamleleri) |
| **İterasyon 28** | Depth 3 | 2 / 4 | 0 | 2 / 4 | **%50.0** | Kırılma (Köşe savunması öğrenildi) |
| **İterasyon 47** | Depth 4 | 2 / 4 | 2 | 0 / 4 | **%75.0** | Direnç (Mağlubiyetsiz seri) |
| **İterasyon 63 (Şampiyon)** | **Depth 4** | **4 / 4** | **0** | **0 / 4** | **%100.0** | **Kusursuz Hakimiyet (Süpürme)** |
| **İterasyon 63 (Şampiyon)** | **Depth 5** | **2 / 2** | **0** | **0 / 2** | **%100.0** | **Bozgun (41 - 23 Skor)** |

---

## 3. Taktiksel Çözümleme: Model Neyi Başardı?

1. **Köşe ve Kenar Bütünlüğü:**
   - İlk turlarda görülen "erken taş çevirme açgözlülüğü" tamamen terk edildi.
   - Model oyun ortasında taş sayısını kasten düşük tutup rakibin hareket alanını kısıtlıyor (mobilite kontrolü).
2. **Minimax Derinlik 5'e Karşı 41 - 23 Üstünlük:**
   - Minimax Derinlik 5 hamle başına ~1.5 - 2 saniye derin hesap yapmasına rağmen, AlphaZero MCTS ile rakibin kenar tuzaklarını boşa çıkardı ve son 15 hamlede tahtayı ele geçirdi.

---

## 4. Sıradaki Hamle (Büyük Final: Edax Turnuva Motoru)

- Connect4'te solver'a ulaştığımız gibi, Othello'da da klasik Minimax defterini **%100 galibiyetle kapattık.**
- Sıradaki nihai sınav: Dünya Othello turnuvalarının referans motoru olan **Edax (Tournament Engine)** karşısına çıkmak.
