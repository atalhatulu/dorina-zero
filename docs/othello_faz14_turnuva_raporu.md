# Othello AlphaZero Faz 14: Çoklu Açılış Turnuva Doğrulama Raporu (v9 Sonuçları)

**Tarih:** 18 Kasım 2026  
**Eğitim Koşusu:** Kaggle v9 (`abdullahtalhatl/dorina-othello-alphazero-edax-v9`)  
**Eski Şampiyon:** İterasyon 125 / İterasyon 143  
**Yeni Şampiyon:** İterasyon 162 (`models/othello/checkpoints/current_best.pt`)  
**Rakip Motor:** Edax Tournament Engine (C binary, GTP protokolü)  
**Test Metodolojisi:** 8 Maçlık Dengeli Açılış Turnuvası (4 Açılış $\times$ Çift Yönlü Renk Eşleşmesi, 200 MCTS Simülasyonu)

---

## 1. Yönetici Özeti ve Temel Bulgular

1. **Edax Veto Kalkanı Kusursuz Çalıştı:**  
   - Kaggle v9 süresince İterasyon 144 (%62.5 self-play) ve İterasyon 154 (%56.2 self-play) kendi içindeki maçları kazanarak terfi talep etti.  
   - Ancak eğitim motoruna eklediğimiz **Edax Veto Gating**, bu modellerin Edax karşısında sırasıyla %12.5 ve %25.0 aldığını tespit etti; terfileri **veto etti** ve ağırlıkları sıfırlayarak modelin "iç havuzda sahte galibiyetlerle unutma/ezber çukuruna düşmesini" engelledi.
   - Sınavı sadece **İterasyon 145 (%75.0 Edax L1)** ve **İterasyon 162 (%75.0 Edax L1)** geçerek terfi aldı.

2. **Gerçek Çoklu Açılış Doğrulaması:**  
   - Geçmişte standart boş tahtada tek açılış oynandığı için model tek bir varyantı ezberleyebiliyordu.
   - `STANDARD_OTHELLO_OPENINGS` (Standart, `d3 c3`, `d3 c5`, `d3 e3`) ile yapılan 8 maçlık genişletilmiş turnuvada **İterasyon 162**, eski referansımız olan **İterasyon 125'i** geride bıraktı:
     - **İterasyon 125 (Eski):** Edax Seviye 1'e karşı 3 / 8 galibiyet (**%37.5**)
     - **İterasyon 162 (Yeni):** Edax Seviye 1'e karşı 4 / 8 galibiyet (**%50.0**)

3. **Edax Seviye 2 Direnci:**  
   - İterasyon 162, çok daha derin arama yapan Edax Seviye 2'ye karşı 8 açılışlık seride 1 Galibiyet (`d3 c5` Beyaz: 36 - 28) ve 1 Beraberlik (`d3 e3` Beyaz: 32 - 32) almayı başardı.

---

## 2. Detaylı Karşılaştırma Matrisi (8 Açılışlık Seri, 200 Simülasyon)

### A. Edax Seviye 1 Karşısındaki Durum

| Açılış | Renk | İterasyon 125 Skoru | İterasyon 162 Skoru (Yeni Şampiyon) |
| :--- | :--- | :---: | :---: |
| **Standart** | Siyah (P1) | 25 - 39 (Edax) | 2 - 60 (Edax) |
| **Standart** | Beyaz (P2) | 16 - 48 (Edax) | 4 - 60 (Edax) |
| **d3 c3** | Siyah (P1) | 25 - 39 (Edax) | 2 - 60 (Edax) |
| **d3 c3** | Beyaz (P2) | **46 - 18 (MODEL)** | **35 - 29 (MODEL)** |
| **d3 c5** | Siyah (P1) | **41 - 23 (MODEL)** | 4 - 60 (Edax) |
| **d3 c5** | Beyaz (P2) | 31 - 33 (Edax) | **33 - 31 (MODEL)** |
| **d3 e3** | Siyah (P1) | 5 - 59 (Edax) | **48 - 16 (MODEL)** |
| **d3 e3** | Beyaz (P2) | **36 - 28 (MODEL)** | **36 - 28 (MODEL)** |
| **TOPLAM SKOR** | | **3 / 8 (%37.5)** | **4 / 8 (%50.0)** |

### B. İterasyon 162 vs Edax Seviye 2 (Zorluk Basamağı)

| Açılış | Renk | Skor (Model - Edax) | Kazanan | Hamle Sayısı |
| :--- | :--- | :---: | :---: | :---: |
| **Standart** | Siyah (P1) | 15 - 49 | EDAX | 60 |
| **Standart** | Beyaz (P2) | 1 - 63 | EDAX | 65 |
| **d3 c3** | Siyah (P1) | 26 - 37 | EDAX | 61 |
| **d3 c3** | Beyaz (P2) | 14 - 50 | EDAX | 61 |
| **d3 c5** | Siyah (P1) | 12 - 52 | EDAX | 62 |
| **d3 c5** | Beyaz (P2) | **36 - 28** | **MODEL** | 60 |
| **d3 e3** | Siyah (P1) | 22 - 42 | EDAX | 60 |
| **d3 e3** | Beyaz (P2) | **32 - 32** | **BERABERE** | 60 |
| **TOPLAM SKOR** | | **1G - 1B - 6M (%18.8 Puan)** | | |

---

## 3. Zayıflık ve Taktiksel Teşhis (Eleştiri Protokolü)

1. **Siyah (P1) Erken Oyun Dağılması:**  
   Model Beyaz (P2) ile oynarken taş stabilitesini ve köşe kontrolünü koruyup Seviye 2'yi bile devirebilirken (`36 - 28`), Siyah başladığı oyunlarda hamle sayısı arttıkça tahta kontrolünü aniden Edax'a teslim ediyor (2-60 ve 4-60 gibi ezici mağlubiyetler).
2. **Standart Açılış Ezberi / Dar Görüşlülük:**  
   Model `d3 e3` ve `d3 c5` gibi yan açılışlarda son derece dirençli oynarken, en popüler ve bilinen "Standart" başlangıçta hem Siyah hem Beyaz ile çöktü. Modelin self-play sırasında standart açılışta yerel bir hata döngüsüne girdiği anlaşılıyor.

---

## 4. Sıradaki Adım (Faz 15 Önerisi)

- **Açılış Havuzu Çeşitlendirmesi (Self-play Opening Randomization):**  
  Self-play oyunlarının ilk 2-4 hamlesini sadece softmax sıcaklığı ile değil, `STANDARD_OTHELLO_OPENINGS` havuzundan rastgele bir açılış pozisyonu yükleyerek başlatmak. Bu, modelin standart başlangıçtaki ezber körlüğünü kıracaktır.
- **Edax Curriculum Oranı ve Derinliği:**  
  Model Seviye 1'de artık %50 tabanına ulaştı. Curriculum hibrit oranını %25'e çıkarıp L2 ağırlığını %70, L1 ağırlığını %30 yaparak Seviye 2'ye geçişi hızlandırmak.
