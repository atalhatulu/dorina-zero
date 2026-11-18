# Othello AlphaZero Turnuva Benchmark Raporu (Faz 11 - 35 İterasyonluk Seviye 2 Eğitimi)

**Tarih:** 18 Kasım 2026  
**Eğitim Paketi:** 35 İterasyon (İter 126 -> İter 160)  
**Curriculum:** Edax Seviye 2 (%15 oran)  
**MCTS:** 200 Simülasyon  

---

## 1. Yönetici Özeti ve Teşhis

35 iterasyonluk uzun koşuda model toplam 15 terfi alarak `current_best.pt` ağırlıklarını **İterasyon 158**'e taşımıştır.

Ancak derin analizde kritik bir dinamik saptanmıştır:
1. **Edax Seviye 2'ye Karşı Gelişme:**
   - İterasyon 125, Seviye 2'ye karşı Beyaz ile 11-53 kaybederken,
   - **İterasyon 158 Beyaz ile taş sayısını 26'ya çıkardı (26 - 38).** Oyunun son 10 hamlesine kadar Edax Seviye 2 ile kafa kafaya mücadele etti.
2. **Unutma (Catastrophic Forgetting) Sinyali:**
   - Model sürekli Seviye 2'nin agresif köşe ve derin taktiklerine maruz kaldığı için `value_loss` 0.25'ten 0.33 bandına yükseldi.
   - Bu durum modelin Edax Seviye 1'e karşı sahip olduğu %100'lük mutlak disiplini kısmen gevşetti (İter 135: 29-35, İter 158: 24-40).

---

## 2. Karşılaşma Skor Tablosu

### A. Edax Seviye 2 Karşılaşması (İterasyon 158)
- **Skor:** 0 / 4 (%0.0)
- **Beyaz (P2):** Model 26 - 38 Edax (61 hamle) - *Gözle görülür direnç artışı*
- **Siyah (P1):** Model 7 - 57 Edax (62 hamle) - *Açılışta köşe kaybı*

### B. Edax Seviye 1 Karşılaşması (İterasyon 158 & 135)
- **İterasyon 135:** Model 29 - 35 Edax (Kafa kafaya mücadele)
- **İterasyon 158:** Model 24 - 40 Edax

---

## 3. Mühendislik Çözümü ve Yol Haritası

Bir motor seviyesinden diğerine zıplarken önceki seviyeyi unutmamak için tek tip Seviye 2 beslemesi yerine:
- **Hibrit Müfredat (%10 Seviye 1 + %10 Seviye 2):** Model hem temel evaporasyon reflekslerini taze tutacak hem de Seviye 2'nin derin varyasyonlarına adapte olacak.
- **Sağlam Kale Güvencesi:** İterasyon 125 ağırlıkları `checkpoints/iter_125.pt` altında saklıdır; gerektiğinde baz model olarak referans alınabilir.
