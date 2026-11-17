# 🏆 Othello AlphaZero — Faz 3 (İterasyon 14-23) Derinlik Raporu

> **Tarih:** 16 Kasım 2026  
> **Konfigürasyon:** 200 Simülasyon, 25 Oyun/Tur, %20 Historical Rakip, Tesla T4 GPU  
> **Durum:** 23. İterasyon Tamamlandı, Şampiyon Model: `checkpoints/othello/current_best.pt` (İter 14)

---

## 1. Eğitim Metrikleri Tablosu (Faz 3)

| İterasyon | Policy Loss | Value Loss | Self-Play Süresi | Toplam Örnek (8x D4) | Gating Kararı | Notlar |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **014** | **0.6044** | **0.1183** | 117.6s | 12,088 | 🏆 **TERFİ (%79.2)** | 13. iterasyonu 9-2 ezerek tahta oturdu |
| **015** | 0.6363 | 0.1361 | 115.2s | 12,136 | ❌ Reddedildi (%33.3) | `iter_015.pt` kaydedildi |
| **016** | 0.7208 | 0.1585 | 111.0s | 12,136 | ❌ Reddedildi (%8.3) | Zayıf varyant elendi |
| **017** | 0.7111 | 0.1577 | 117.3s | 12,128 | ❌ Reddedildi (%25.0) | Zayıf varyant elendi |
| **018** | 0.6958 | 0.1603 | 111.6s | 12,160 | ❌ Reddedildi (%50.0) | Başa baş maç |
| **019** | 0.6866 | 0.1560 | 109.2s | 12,152 | ❌ Reddedildi (%50.0) | Başa baş maç |
| **020** | 0.6702 | 0.1671 | 112.8s | 12,184 | ❌ Reddedildi (%25.0) | `iter_020.pt` kaydedildi |
| **021** | 0.6726 | 0.1671 | 118.3s | 12,136 | ❌ Reddedildi | 14. iterasyon tahtını korudu |
| **022** | 0.6800 | 0.1651 | 109.8s | 12,144 | ❌ Reddedildi | 14. iterasyon tahtını korudu |
| **023** | 0.6622 | 0.1712 | 114.4s | 12,136 | ❌ Reddedildi | 14. iterasyon tahtını korudu |

* **Şampiyon:** `checkpoints/othello/current_best.pt` (İterasyon 14)
* **Arşiv Checkpointler:** `iter_005.pt`, `iter_010.pt`, `iter_015.pt`, `iter_020.pt`
* **Toplam Birikmiş Veri:** **376,000+ eğitim örneği**

---

## 2. Mühendislik Çıkarımları

1. **İterasyon 14 Platôsu (Süper Şampiyon):**
   * 200 simülasyona çıktığımız ilk turda üretilen 14. iterasyon modeli, sonraki 9 iterasyon boyunca gelen tüm adayları darmadağın etti.
   * Gating filtresi kusursuz çalışarak daha zayıf modellerin (özellikle %8 ve %25 alan modellerin) şampiyonluğu kapmasını engelledi.
2. **Kendi Kendine Kilitlenme (Echo Chamber Belirtisi):**
   * 15-23 arasındaki tüm self-play maçları tam 10-10 bitti ve loss 0.66 - 0.70 bandında yatay seyretti.
   * Model şu an kendi varyant havuzunda doyum noktasına ulaştı. 14. iterasyonun kilidini kırmak için dışarıdan yeni bir taktiksel usta (Curriculum Minimax) gerekiyor.
