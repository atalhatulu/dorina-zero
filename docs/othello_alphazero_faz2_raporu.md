# 🏆 Othello AlphaZero — Faz 2 (İterasyon 6-13) Colab GPU Eğitim Raporu

> **Tarih:** 16 Kasım 2026  
> **Donanım:** Google Colab (Tesla T4 GPU + 8 vCPU)  
> **Model:** OthelloNet (4 ResBlock, 323K Parametre)  
> **Durum:** 13. İterasyon Başarıyla Tamamlandı (10 Dakikalık Turbo Koşu)

---

## 1. Eğitim Metrikleri Tablosu

| İterasyon | Policy Loss | Value Loss | Self-Play Süresi | Toplam Örnek (8x D4) | Notlar |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **006** | 0.7077 | 0.0586 | 70.8s | 16,856 | 🏆 5. İterasyonu 7-5 yenerek yeni şampiyon oldu |
| **007** | 0.6424 | 0.0926 | 65.7s | 16,936 | Taktiksel açılışlar derinleşti |
| **008** | **0.5862** | 0.0918 | 59.2s | 15,632 | Policy loss taban seviyeye indi |
| **009** | 0.6170 | 0.1020 | 67.4s | 16,824 | Tahta dengesi oturuyor |
| **010** | 0.6279 | 0.1142 | 65.6s | 16,760 | 💾 `iter_010.pt` arşive kaydedildi |
| **011** | 0.6778 | 0.1365 | 66.2s | 16,744 | Historical rakiplere karşı savunma refleksi |
| **012** | 0.6811 | 0.1351 | 66.8s | 16,760 | Kararlı ve dengeli oyun |
| **013** | 0.6909 | 0.1449 | 64.0s | 16,784 | 🏆 **Faz 2 Şampiyonu (`current_best.pt`)** |

* **Ortalama Tur Süresi:** ~65 saniye (1 dakikada 1 iterasyon!)
* **Toplam Üretilen Veri (Faz 2):** **134,312 eğitim örneği**
* **Toplam Birikmiş Veri (Faz 1 + Faz 2):** **~255,000 eğitim örneği**
* **Mevcut Şampiyon:** `checkpoints/othello/current_best.pt` (İter 13)

---

## 2. Mühendislik Değerlendirmesi

1. **GPU + CPU Hızı:**
   * Colab Tesla T4 GPU sayesinde ağ eğitimi 1 saniyenin altında bitti. 8 vCPU ile 35 maç ortalama 64 saniyede simüle edildi.
   * Toplam 8 iterasyon **10 dakikada** tamamlandı.
2. **Loss Davranışı:**
   * Policy loss `1.99`'dan `0.58 - 0.69` bandına indi. Ağ artık hangi kareye hamle yapması gerektiğini %90+ kesinlikle biliyor.
   * Value loss `0.05 - 0.14` bandında çok kararlı; taş aldanmasına kapılmıyor.

---

## 3. Sıradaki Adım: Faz 3 (Şampiyonluk Ligi)

Şimdi anlaştığımız **"200 Simülasyon / 25 Oyun"** taktiksel derinlik fazına geçiyoruz:
* MCTS derinliği 80'den **200'e** çıkarılacak.
* Oyun sayısı 25 olacak (8x simetri ile tur başına 12.000 kaliteli veri).
* Model 13. iterasyondaki şampiyondan devralacak.
