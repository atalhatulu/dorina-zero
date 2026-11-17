# Othello AlphaZero Faz 7 Adli Analiz Raporu: İterasyon 94 Teşhisi ve Çözüm

**Tarih:** 17 Kasım 2026  
**İncelenen Model:** `models/othello/checkpoints/current_best.pt` (İterasyon 94)  
**Kıyaslanan Model:** `iter_078.pt` (İterasyon 78 - Edax'ı 35-29 yenen model)  

---

## 1. Kritik Göstergeler: Ne Oldu?

Kaggle üzerindeki 16 iterasyonluk (79 $\rightarrow$ 94) Edax koşusu sonrasında iki farklı test yapıldı:
1. **Edax Level 1 Testi:** 0 / 4 Galibiyet (8-56, 11-53 yenilgiler).
2. **Minimax Depth 4 Kontrol Testi:** 0 / 4 Galibiyet (18-46, 24-40 yenilgiler).

**Çıkarım:** İterasyon 78'de Edax'ı deviren, Minimax'i 4'te 4 süpüren model, 94. iterasyona gelindiğinde taktiksel bir gerileme (regresyon) yaşadı.

---

## 2. Kök Neden Analizi (Root Cause)

Kaggle logları ve `core/self_play.py` satır satır incelendiğinde 2 temel neden tespit edildi:

1. **"One-Hot Dirac Dağılımı" Şoku (Curriculum Overfitting):**
   - `core/self_play.py` satır 206-207:
     ```python
     policy = np.zeros(game.action_space_size, dtype=np.float32)
     policy[action] = 1.0
     history.append((encoded, policy, current_player))
     ```
   - Normal self-play'de MCTS 200 simülasyon yaparak **yumuşak (smooth) bir olasılık dağılımı** üretir (örneğin %40 bir hamle, %25 başka hamle).
   - Edax maçlarında ise botun tek bir hamlesine %100 (`1.0`) olasılık atanarak buffer'a basıldı.
   - Bu durum Cross-Entropy kaybında gradyan patlamasına yol açtı; model MCTS ile alternatif aramayı bırakıp sadece Edax'ın tekil hamlelerini taklit etmeye (overfit olmaya) çalıştı.
2. **Value Loss Artışı:**
   - Loglarda Value Loss 0.14'ten **0.30'a fırladı**. Model oyunun kimin kazanacağını kestiremez hale geldi.

---

## 3. Kurtarma ve Düzeltme Planı

1. **En Güçlü Kaleye Geri Çekilme (Rollback to Champion 78):**
   - Elimizde Edax'ı 35-29 yenen sapasağlam **`iter_078.pt`** checkpoint'i duruyor.
   - `current_best.pt` olarak tekrar `iter_078.pt` atanacak.
2. **Curriculum Dağılımının Düzeltilmesi (MCTS-guided Curriculum):**
   - Bot hamle yaptığında doğrudan %100 one-hot basmak yerine, botun seçtiği hamleye %70, geçerli diğer hamlelere %30 dağıtılacak (veya bot hamlesinden sonra kısa bir MCTS smoothing uygulanacak).
3. **Curriculum Dozajının Dengelenmesi:**
   - Oran %25'ten **%15** seviyesine çekilecek. Modelin asıl omurgası kendi MCTS kendini-oynama (self-play) verisi olmalı; bot sadece taktiksel bir aşı olarak kalmalıdır.
