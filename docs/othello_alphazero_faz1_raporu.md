# 🏆 Othello AlphaZero — Faz 1 (İterasyon 1-5) Başlangıç Raporu

> **Tarih:** 15 Kasım 2026  
> **Oyun:** Othello (Reversi) 8x8  
> **Faz:** Faz 1 (Tabula Rasa - Sıfırdan Keşif)  
> **Sonuç:** 5 İterasyon Başarıyla Tamamlandı, 5'te 5 Terfi (%100 Gating Başarısı)

---

## 1. Genel Metrik Özeti

| İterasyon | Policy Loss | Value Loss | P1 Galibiyet | P2 Galibiyet | Beraberlik | Üretilen Veri (8x D4) | Gating Skoru | Terfi Durumu |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **001** | 3.3562 | 0.3318 | 23 | 25 | 2 | 24,088 | %66.7 (15-7-2) | 🏆 Terfi |
| **002** | 2.6963 | 0.2800 | 22 | 24 | 4 | 24,136 | %81.2 (18-3-3) | 🏆 Terfi |
| **003** | 2.2857 | 0.2960 | 23 | 26 | 1 | 24,184 | %68.8 (16-7-1) | 🏆 Terfi |
| **004** | 2.1092 | 0.3066 | 27 | 21 | 2 | 24,144 | %62.5 (13-7-4) | 🏆 Terfi |
| **005** | **1.9970** | **0.3014** | 20 | 27 | 3 | 24,144 | **%68.8 (16-7-1)** | 🏆 **Terfi & Arşiv** |

* **Toplam Oynanan Oyun:** 250 maç
* **D4 ile Üretilen Eğitim Örneği:** **120,696 tensör**
* **Arşiv Checkpoint:** `checkpoints_othello/iter_005.pt`
* **Aktif Şampiyon:** `checkpoints_othello/current_best.pt`

---

## 2. Mühendislik Çıkarımları

1. **Loss Eğrisi Kararlılığı:**
   * Policy loss 4.15'ten başlayıp 5. iterasyon sonunda **1.9970'e** indi. Ağ, tahta koordinatlarını ve temel taş çevirme hamlelerini başarıyla kavradı.
2. **First-Player / Second-Player Dengesi:**
   * P1 (Siyah) galibiyetleri: 115 (%46.0)
   * P2 (Beyaz) galibiyetleri: 123 (%49.2)
   * Beraberlikler: 12 (%4.8)
   * Connect 4'teki gibi P1 lehine ezici bir asimetri oluşmadı; oyun dengeli bir güç eğrisi sergiliyor.
3. **Gating Filtresi:**
   * Her iterasyonda yeni model, önceki şampiyonu tarafsız açılış havuzunda (%55 eşik şartıyla) net bir üstünlükle (ortalama %69.6 puanla) mağlup etti.

---

## 3. Sıradaki Adım (Faz 2 Önerisi)

Model artık temel taş koymayı öğrendiğine göre taktiksel ustalığa geçebiliriz:
1. **İterasyon 6-15:** %80 Self-Play + %20 Historical Rakip (`iter_005.pt` ve geçmiş şampiyonlar).
2. Othello'ya özel Heuristic Minimax (köşe/mobilite ağırlıklı bot) entegrasyonu ile curriculum eğitimi.
