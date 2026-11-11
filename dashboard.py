"""dorina4 - Connect4 AlphaZero Canlı Eğitim Dashboard'u."""
import os
import json
import glob
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Sayfa yapılandırması
st.set_page_config(
    page_title="dorina4 — Connect4 AlphaZero",
    page_icon="🎮",
    layout="wide",
)

LOGS_DIR = "logs"
CHECKPOINTS_DIR = "checkpoints"
ITERATIONS_CSV = os.path.join(LOGS_DIR, "iterations.csv")
EVAL_CSV = os.path.join(LOGS_DIR, "eval.csv")
COVERAGE_JSON = os.path.join(LOGS_DIR, "coverage.json")


def load_iterations_data():
    """iterations.csv dosyasını oku."""
    # Her okumada log dosyasından en güncel CSV'yi oluştur
    try:
        import parse_log_to_csv
    except Exception:
        pass
    if os.path.exists(ITERATIONS_CSV):
        try:
            df = pd.read_csv(ITERATIONS_CSV)
            if not df.empty:
                return df
        except Exception:
            return None
    return None


def load_coverage_data():
    """coverage.json dosyasını oku."""
    if os.path.exists(COVERAGE_JSON):
        try:
            with open(COVERAGE_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_checkpoints():
    """checkpoints klasöründeki .pt dosyalarını listele."""
    files = glob.glob(os.path.join(CHECKPOINTS_DIR, "iter_*.pt"))
    ckpts = []
    for f in sorted(files, reverse=True):
        stat = os.stat(f)
        ckpts.append({
            "Dosya": os.path.basename(f),
            "Boyut (MB)": round(stat.st_size / (1024 * 1024), 2),
            "Son Güncelleme": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return pd.DataFrame(ckpts)


# --- Yan Panel (Sidebar) ---
st.sidebar.header("⚙️ Kontroller")

stage_mode = st.sidebar.radio(
    "Görüntülenen Aşama",
    options=["Aşama 3 (Kararlı 20 İter - Aktif)", "Aşama 2 (116K Model - Arşiv)", "Aşama 1 (465K Model - Arşiv)"],
    index=0,
)

if "Aşama 3" in stage_mode:
    CHECKPOINTS_DIR = os.path.join("checkpoints", "connect4")
    ITERATIONS_CSV = os.path.join("logs_v3", "iterations_v3.csv")
    ACTIVE_LOG_FILE = "training_20iter_stable.log"
    total_target = 20
elif "Aşama 2" in stage_mode:
    CHECKPOINTS_DIR = "checkpoints_v2"
    ITERATIONS_CSV = os.path.join("logs_v2", "iterations_v2.csv")
    ACTIVE_LOG_FILE = "training_v2.log"
    total_target = 30
else:
    CHECKPOINTS_DIR = "checkpoints"
    ITERATIONS_CSV = os.path.join("logs", "iterations.csv")
    ACTIVE_LOG_FILE = "training_30iter.log"
    total_target = 30

if st.sidebar.button("🔄 Şimdi Yenile"):
    st.rerun()

refresh_option = st.sidebar.selectbox(
    "Otomatik Yenileme Aralığı",
    options=["5s", "10s", "30s", "Kapalı"],
    index=1,
)

refresh_seconds = {"5s": 5, "10s": 10, "30s": 30, "Kapalı": None}[refresh_option]

# --- Ana Başlık ---
st.title("🎮 dorina4 — Connect4 AlphaZero Canlı İzleme")

df_iter = load_iterations_data()
cov_data = load_coverage_data()

# --- 2. Üst Metrikler (4 sütun) ---
c1, c2, c3, c4 = st.columns(4)

if df_iter is not None and not df_iter.empty:
    last_row = df_iter.iloc[-1]
    prev_row = df_iter.iloc[-2] if len(df_iter) > 1 else None

    c1.metric("Son İterasyon", int(last_row["iteration"]))

    # Policy loss ve delta (Loss düştüğünde yeşil/iyi, arttığında kırmızı/kötü)
    if pd.notna(last_row.get("policy_loss")):
        pol_val = last_row["policy_loss"]
        pol_loss = f"{pol_val:.4f}"
        if prev_row is not None and pd.notna(prev_row.get("policy_loss")):
            pol_delta = f"{(pol_val - prev_row['policy_loss']):+.4f}"
        else:
            pol_delta = None
        c2.metric("Policy Loss (Son)", pol_loss, delta=pol_delta, delta_color="inverse")
    else:
        c2.metric("Policy Loss (Son)", "N/A")

    # Value loss ve delta (Loss düştüğünde yeşil/iyi, arttığında kırmızı/kötü)
    if pd.notna(last_row.get("value_loss")):
        val_val = last_row["value_loss"]
        val_loss = f"{val_val:.4f}"
        if prev_row is not None and pd.notna(prev_row.get("value_loss")):
            val_delta = f"{(val_val - prev_row['value_loss']):+.4f}"
        else:
            val_delta = None
        c3.metric("Value Loss (Son)", val_loss, delta=val_delta, delta_color="inverse")
    else:
        c3.metric("Value Loss (Son)", "N/A")
else:
    c1.metric("Son İterasyon", "Henüz Yok")
    c2.metric("Policy Loss (Son)", "N/A")
    c3.metric("Value Loss (Son)", "N/A")

if cov_data:
    cov_pct = f"{cov_data.get('coverage_pct', 0):.6f}%"
    c4.metric("Tahta Kapsamı (Coverage)", cov_pct, help=cov_data.get("coverage_str", ""))
else:
    c4.metric("Tahta Kapsamı (Coverage)", "N/A")

# --- Eğitim Durumu, İlerleme ve ETA ---
st.subheader("⏳ Eğitim Durumu ve İlerleme")
total_target = 30  # Hedef iterasyon sayısı

# Eğitimin aktif olup olmadığını ilgili log dosyasının son değişiklik zamanından anla
is_training_active = False
target_log = ACTIVE_LOG_FILE
if os.path.exists(target_log):
    mtime = os.path.getmtime(target_log)
    # Son 3 dakika içinde log dosyasına yazılmışsa eğitim aktiftir
    if (time.time() - mtime) < 180:
        is_training_active = True

# Checkpoint dosyalarının gerçek disk zaman damgalarını al
ckpt_pattern = os.path.join(CHECKPOINTS_DIR, "v2_iter_*.pt" if "Aşama 2" in stage_mode else "iter_*.pt")
ckpt_files = sorted(glob.glob(ckpt_pattern))
ckpt_times = []
for cf in ckpt_files:
    try:
        base = os.path.basename(cf).replace(".pt", "")
        it_num = int(base.replace("v2_iter_", "").replace("iter_", ""))
        ckpt_times.append((it_num, os.path.getmtime(cf)))
    except Exception:
        pass
ckpt_times.sort(key=lambda x: x[0])

# İterasyonlar arası gerçek süreleri hesapla
iter_durations = []
for i in range(1, len(ckpt_times)):
    diff = ckpt_times[i][1] - ckpt_times[i-1][1]
    if 10 < diff < 900:  # makul bir iterasyon süresi penceresi
        iter_durations.append(diff)

avg_iter_sec = (sum(iter_durations[-5:]) / len(iter_durations[-5:])) if iter_durations else 140.0

current_iter = ckpt_times[-1][0] if ckpt_times else (int(df_iter.iloc[-1]["iteration"]) if df_iter is not None and not df_iter.empty else 0)

# Eğitim bittiyse (iter_30 tamamlandıysa veya süreç sonlandıysa)
if current_iter >= total_target or not is_training_active:
    col_prog, col_status = st.columns([3, 1])
    with col_prog:
        st.progress(1.0)
        st.caption(f"🏁 **Eğitim Tamamlandı!** Toplam **{current_iter}/{total_target}** iterasyon checkpoint'i hazır.")
    with col_status:
        st.info("🏁 Tamamlandı")
else:
    # Aktif iterasyon içi geçen süreyi son checkpoint zamanından hesapla
    last_ckpt_time = ckpt_times[-1][1] if ckpt_times else time.time()
    elapsed_current = max(0.0, time.time() - last_ckpt_time)
    iter_progress = min(elapsed_current / avg_iter_sec, 0.95)

    # Genel zaman/iterasyon bazlı hassas ilerleme
    base_progress = current_iter / total_target
    fine_progress = min(max(base_progress + (iter_progress / total_target), 0.0), 0.99)

    remaining_iters = total_target - current_iter
    remaining_sec = max(0.0, (remaining_iters * avg_iter_sec) - elapsed_current)
    remaining_min = remaining_sec / 60.0

    col_prog, col_status = st.columns([3, 1])
    with col_prog:
        st.progress(fine_progress)
        st.caption(
            f"🚀 Genel İlerleme: **%{fine_progress * 100:.1f}** | "
            f"İterasyon {current_iter + 1}/{total_target} içi: **%{iter_progress * 100:.0f}** ({elapsed_current:.0f}s / ~{avg_iter_sec:.0f}s) | "
            f"Kalan: **~{remaining_sec:.0f}s** (~{remaining_min:.1f} dk)"
        )
    with col_status:
        st.success("🟢 Eğitim Sürüyor")

st.divider()

# --- Grafikler ve Tablolar ---
if df_iter is None or df_iter.empty:
    st.info("ℹ️ Henüz eğitim verisi bulunamadı (`logs/iterations.csv`). Eğitim başladığında burası otomatik güncellenecektir.")
else:
    col_left, col_right = st.columns(2)

    # 3. Loss Grafiği
    with col_left:
        st.subheader("📉 Kayıp (Loss) Trendi")
        fig_loss = go.Figure()
        if "policy_loss" in df_iter.columns:
            fig_loss.add_trace(go.Scatter(
                x=df_iter["iteration"], y=df_iter["policy_loss"],
                mode="lines+markers", name="Policy Loss", line=dict(color="#1f77b4", width=2)
            ))
        if "value_loss" in df_iter.columns:
            fig_loss.add_trace(go.Scatter(
                x=df_iter["iteration"], y=df_iter["value_loss"],
                mode="lines+markers", name="Value Loss", line=dict(color="#ff7f0e", width=2)
            ))
        fig_loss.update_layout(
            xaxis_title="İterasyon", yaxis_title="Loss",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # 4. Kazanma Dağılımı (Stacked Bar - Son 20)
    with col_right:
        st.subheader("🏆 Son 20 İterasyon Kazanma Dağılımı")
        recent_df = df_iter.tail(20)
        fig_wins = go.Figure()
        fig_wins.add_trace(go.Bar(
            x=recent_df["iteration"], y=recent_df.get("p1_wins", 0),
            name="1. Oyuncu (Kırmızı)", marker_color="#d62728"
        ))
        fig_wins.add_trace(go.Bar(
            x=recent_df["iteration"], y=recent_df.get("p2_wins", 0),
            name="2. Oyuncu (Sarı)", marker_color="#bcbd22"
        ))
        fig_wins.add_trace(go.Bar(
            x=recent_df["iteration"], y=recent_df.get("draws", 0),
            name="Beraberlik (Draw)", marker_color="#7f7f7f"
        ))
        fig_wins.update_layout(
            barmode="stack", xaxis_title="İterasyon", yaxis_title="Oyun Sayısı",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_wins, use_container_width=True)

    col_mid1, col_mid2 = st.columns(2)

    # 5. Ortalama Hamle Sayısı
    with col_mid1:
        st.subheader("⏱️ Ortalama Oyun Uzunluğu (Hamle)")
        if "avg_moves" in df_iter.columns:
            fig_moves = px.line(
                df_iter, x="iteration", y="avg_moves", markers=True,
                labels={"iteration": "İterasyon", "avg_moves": "Ort. Hamle"}
            )
            fig_moves.update_traces(line_color="#2ca02c")
            fig_moves.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_moves, use_container_width=True)

    # 6. Benzersiz Pozisyon Sayısı (Coverage Trendi / Durumu)
    with col_mid2:
        st.subheader("🧩 Pozisyon Havuzu ve Kapsam")
        if cov_data:
            st.write(f"**Benzersiz Ziyaret Edilen Pozisyon:** `{cov_data.get('unique_positions', 0):,}`")
            st.write(f"**Toplam Tromp Uzayı:** `{cov_data.get('total_positions', 0):,}`")
            st.write(f"**Kapsam Oranı:** `{cov_data.get('coverage_pct', 0):.8f}%`")
            st.progress(min(cov_data.get("unique_positions", 0) / 1_000_000, 1.0))
            st.caption("1M pozisyonluk ilk keşif hedefi baz alınmıştır.")
        else:
            st.write("Coverage verisi henüz oluşmadı.")

    st.divider()

    # 7. Son 10 İterasyon Tablosu
    st.subheader("📋 Son İterasyonlar")
    st.dataframe(df_iter.tail(10).iloc[::-1], use_container_width=True)

# 8. Checkpoint Listesi
st.divider()
st.subheader("💾 Kaydedilen Checkpoint'ler")
df_ckpts = get_checkpoints()
if not df_ckpts.empty:
    st.dataframe(df_ckpts, use_container_width=True)
else:
    st.write("Henüz checkpoint kaydedilmedi.")

# Otomatik yenileme döngüsü
if refresh_seconds:
    time.sleep(refresh_seconds)
    st.rerun()
