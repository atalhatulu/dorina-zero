# dorina-zero 🎮

AlphaZero tabanlı, kendi kendine öğrenen (self-play) ve Monte Carlo Ağaç Araması (MCTS) destekli **Connect 4** yapay zekası.

---

## 🚀 Özellikler

- **AlphaZero Mimarisi:** Politika (Policy) ve Değer (Value) tahminlerini eş zamanlı üreten çift başlıklı derin evrişimli sinir ağı (Dual-head CNN).
- **MCTS (Monte Carlo Tree Search):** PUCT formülü ve Dirichlet gürültüsü ile dengelenmiş derin taktiksel arama.
- **Canlı Değerlendirme Çubuğu (Evaluation Bar):** Satranç motorlarındakine benzer, GUI üzerinde anlık kazanma ihtimalini ve en iyi hamle önerisini gösteren Pygame arayüzü.
- **Gerçek Zamanlı Dashboard:** Streamlit & Plotly tabanlı canlı eğitim metrikleri, kayıp grafikleri ve kazanma dağılımı takibi.
- **Hafifletilmiş Ağ (Aşama 2):** 465K parametrelik ilk modelden $1 \times 1$ Conv kafalarıyla **116K** parametreye indirilmiş, CPU'da yüksek FPS sağlayan `Connect4NetV2`.
- **Yüksek Performans Hazırlığı:**
  - $O(1)$ Bitboard (`board_fast.py`) ile 20x daha hızlı kural ve galibiyet kontrolü.
  - Transposition Table (LRU Cache) ile tekrarlanan pozisyonlarda MCTS hızlandırması.
  - ONNX Runtime CPU vektör çıkarım entegrasyonu.

---

## 🛠️ Kurulum

```bash
# Depoyu klonlayın
git clone https://github.com/<kullanici_adiniz>/dorina-zero.git
cd dorina-zero

# Sanal ortam oluşturup aktifleştirin
python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
pip install streamlit plotly pandas onnx onnxruntime
```

---

## 🎮 Oynanış (GUI)

### 1. AlphaZero Modeline Karşı Oyna
```bash
python main.py --model checkpoints_v2/v2_iter_010.pt --sims 100
```

### 2. Klasik Minimax Motoruna Karşı Oyna
```bash
python main.py --minimax-depth 4
```

---

## 📊 Canlı Eğitim Dashboard'u

Eğitimi izlemek için canlı dashboard'u başlatın:
```bash
streamlit run dashboard.py
```
Tarayıcınızda `http://localhost:8501` adresine giderek loss değişimlerini, kazanma/beraberlik dağılımını ve kalan tahmini süreyi canlı izleyebilirsiniz.

---

## 🧠 Model Karşılaştırma & Değerlendirme

Checkpoint'leri Minimax motoruna karşı test etmek için:
```bash
python run_eval.py --a checkpoints_v2/v2_iter_010.pt --games 20 --sims 100 --minimax-depth 4
```

---

## 🧪 Testler

Tüm birim testleri çalıştırmak için:
```bash
pytest -v
```

---

## 📄 Lisans
MIT License.
