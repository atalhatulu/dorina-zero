# Dorina4 - Derin Analiz ve Mimari Değerlendirme Raporu

Tarih: 11 Kasım 2026

## 1. Beraberlik (Draw = 127/300, %42) Sorunu
* **Nedenleri:**
  - `dirichlet_epsilon = 0.25`: Connect4 dar bir aksiyon alanına (7 sütun) sahiptir. Satranç/Go gibi geniş tahtalarda 0.25 keşif için iyiyken, 7 sütunlu oyunda %25 gürültü kritik bloklama/kazanma hamlelerini rastgele bozarak tahtanın dolmasına ve yapay beraberliklere yol açar.
  - İlk 10 hamle `temperature=1.0` ile örnekleme yapılması, taşların stratejik kümelenme yerine dağınık yerleşmesine ve kilitlenmeye zemin hazırlamaktadır.
* **Uygulanan Çözüm ve Öneriler:**
  - `dirichlet_epsilon = 0.15` seviyesine çekildi.
  - Alternatif/İleri Adım: Keşif sıcaklığı (`temperature=1.0`) 10. hamle yerine 6. hamleden sonra 0'a indirilebilir.

---

## 2. Değer (Value) Asimetrisi ve Savunma Açığı
* **Durum:** Kazanılacak pozisyonlarda `value = +0.994` iken kaybedilecek pozisyonlarda `value = +0.187` (negatif olmalıydı).
* **Neden:**
  - Kendi kendine oyunda kazanan hep p1 veya p2 olduğu için ve terminal adımlara yakın anlarda kaybetme sinyali MCTS değer yayılımında zayıf kalabilmektedir.
  - Simetrik augmentasyon veri çeşitliliğini artırsa da kayıp pozisyonlarının cezalandırılmasında dengesizlik yaratabilir.
* **Önerilen Müdahale:**
  - Değer kaybı (Value loss) ağırlığını `policy_loss + 1.5 * value_loss` veya `2.0 * value_loss` şeklinde ölçeklemek.
  - Replay buffer içinde mağlubiyetle sonuçlanan oyun adımlarını dengeleyecek ağırlıklı örnekleme (weighted sampling / hard negative mining) eklemek.

---

## 3. Batch MCTS ve Paralelleştirme (CPU Üzerinde)
* **Mevcut Durum:** 11 process, her biri bağımsız olarak `batch_size = 1` ile PyTorch forward çağrısı yapıyor.
* **Batch MCTS (Leaf Parallelization / Virtual Loss):**
  - CPU üzerinde forward maliyeti GPU'ya göre farklıdır. Çoklu process yerine thread + batch forward yapısı kurulursa CPU çekirdekleri tensör çarpımlarında BLAS/OpenMP ile daha iyi doyar.
  - Ancak MCTS leaf parallelization veya batched rollouts kod karmaşıklığını 3 katına çıkarır (asenkron simülasyon kuyrukları, sanal kayıp takibi).
  - Karar: Şu aşamada 11 worker multiprocessing CPU için pratik ve stabil çözümdür; mimari darboğaz çözülmeden batch MCTS eklemek gereksiz risk oluşturur.

---

## 4. FC Katmanı İsrafı ve ResNet Mimarisi
* **Mevcut Yapı:** 465.000 parametrenin %74'ü `nn.Linear(2688, 128)` katmanında bulunmaktadır (`64 * 6 * 7 = 2688`).
* **ResNet + 1x1 Conv Alternatifi:**
  - Conv katmanından sonra `1x1 Conv (64 -> 2)` ile kanal sayısı düşürülüp `2 * 6 * 7 = 84` tensör boyutuyla FC'ye girilirse, parametre sayısı ~465K'dan ~70K seviyesine iner.
  - CPU üzerindeki etkisi: Parametre sayısının düşmesi bellek bant genişliği yükünü azaltır, CPU'da hız kaybı yaşatmaz, aksine forward süresini kısaltabilir.
  - Bir sonraki mimari revizyonda uygulanması tavsiye edilir.

---

## 5. `games.jsonl` ve Log Akışı
* **Eksiklik:** `run_self_play.py` içinde worker'lar `log=False` ile çalıştırılıyor ve dönen veriden yalnızca kümülatif `stats` kaydediliyor; tekil oyun detayları `games.jsonl` dosyasına basılmıyor.
* **Çözüm:** `parallel_self_play` sonuçlarını toplarken `logger.log_game` çağrısı ana döngüye bağlanmalıdır.
