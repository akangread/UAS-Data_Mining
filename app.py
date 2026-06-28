import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clustering Bank Sampah DKI Jakarta",
    page_icon="♻️",
    layout="wide",
)

st.title("♻️ Clustering Bank Sampah DKI Jakarta")
st.markdown("Analisis Spasial menggunakan **DBSCAN** berbasis koordinat geografis per kelurahan.")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["latitude", "longtitude"])
    df["status_kegiatan"] = df["status_kegiatan"].str.strip().str.upper()
    df["is_aktif"] = df["status_kegiatan"] == "AKTIF"
    return df

uploaded = st.sidebar.file_uploader(
    "📂 Upload CSV (opsional – default pakai data bawaan)", type="csv"
)

DEFAULT_CSV = "banksampah_dkijkt.csv"
try:
    if uploaded:
        df = load_data(uploaded)
    else:
        df = load_data(DEFAULT_CSV)
    st.sidebar.success(f"✅ Data dimuat: {len(df):,} baris")
except FileNotFoundError:
    st.error(
        "⚠️ File `banksampah_dkijkt.csv` tidak ditemukan di direktori yang sama dengan `app.py`.\n\n"
        "Silakan upload file CSV melalui sidebar."
    )
    st.stop()

df_aktif = df[df["is_aktif"]].copy()

# ─── SIDEBAR PARAMETER ────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Parameter DBSCAN")

eps_km = st.sidebar.slider(
    "eps (km)", min_value=0.3, max_value=5.0, value=1.0, step=0.1,
    help="Radius maksimum untuk mengelompokkan titik-titik sebagai tetangga."
)
min_samples = st.sidebar.slider(
    "min_samples", min_value=2, max_value=10, value=3, step=1,
    help="Jumlah minimum titik dalam radius eps untuk membentuk klaster."
)

st.sidebar.header("🔍 Opsi Evaluasi Otomatis")
run_grid = st.sidebar.checkbox("Jalankan Grid Search Parameter", value=False)

# ─── PREPROCESSING ─────────────────────────────────────────────────────────────
@st.cache_data
def build_kel(df_aktif: pd.DataFrame) -> pd.DataFrame:
    kel = df_aktif.groupby("kelurahan").agg(
        wilayah=("wilayah", "first"),
        kecamatan=("kecamatan", "first"),
        latitude=("latitude", "mean"),
        longtitude=("longtitude", "mean"),
        total_bank_sampah=("nama_bank_sampah", "count"),
        jumlah_aktif=("is_aktif", "sum"),
        jumlah_tidak_aktif=("is_aktif", lambda x: (~x).sum()),
    ).reset_index()
    kel["pct_aktif"] = (kel["jumlah_aktif"] / kel["total_bank_sampah"] * 100).round(1)
    kel = kel[~kel["wilayah"].str.contains("KAB. ADM. KEP. SERIBU", case=False, na=False)].reset_index(drop=True)
    return kel

kel = build_kel(df_aktif)
coords = kel[["latitude", "longtitude"]].values
coords_rad = np.radians(coords)

# ─── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Statistik", "📈 K-Distance Plot", "🔬 Evaluasi Parameter",
    "🗺️ Peta Interaktif", "📋 Tabel Hasil"
])

# ─── TAB 1: STATISTIK ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Statistik Umum")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", f"{len(df):,}")
    col2.metric("Bank Sampah Aktif", f"{df['is_aktif'].sum():,}")
    col3.metric("Total Kelurahan", f"{len(kel):,}")
    col4.metric("Total Wilayah", f"{kel['wilayah'].nunique()}")

    st.divider()
    st.subheader("Distribusi Per Wilayah")

    wil = kel.groupby("wilayah").agg(
        total_bank_sampah=("total_bank_sampah", "sum"),
        jumlah_aktif=("jumlah_aktif", "sum"),
        jumlah_tidak_aktif=("jumlah_tidak_aktif", "sum"),
        n_kelurahan=("kelurahan", "count"),
    ).reset_index()
    wil["wilayah_short"] = (
        wil["wilayah"].str.replace("KOTA ADM. ", "").str.replace("KAB. ADM. ", "")
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    x = np.arange(len(wil))
    w = 0.35
    axes[0].bar(x - w / 2, wil["jumlah_aktif"], w, label="Aktif", color="#2ecc71")
    axes[0].bar(x + w / 2, wil["jumlah_tidak_aktif"], w, label="Tidak Aktif", color="#e74c3c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(wil["wilayah_short"], rotation=30, ha="right", fontsize=9)
    axes[0].set_title("Jumlah Bank Sampah Aktif vs Tidak Aktif per Wilayah", fontweight="bold")
    axes[0].set_ylabel("Jumlah Bank Sampah")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    kel_sorted = kel.sort_values("total_bank_sampah", ascending=False).head(20)
    colors_bar = ["#2ecc71" if p >= 50 else "#e74c3c" for p in kel_sorted["pct_aktif"]]
    axes[1].barh(kel_sorted["kelurahan"], kel_sorted["total_bank_sampah"], color=colors_bar)
    axes[1].set_xlabel("Total Bank Sampah")
    axes[1].set_title(
        "Top 20 Kelurahan – Bank Sampah Terbanyak\n(hijau=mayoritas aktif, merah=mayoritas tidak aktif)",
        fontweight="bold",
    )
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].invert_yaxis()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ─── TAB 2: K-DISTANCE PLOT ───────────────────────────────────────────────────
with tab2:
    st.subheader("K-Distance Plot untuk Tuning eps")
    st.info(
        "Cari **titik siku (elbow)** pada grafik di bawah. "
        "Nilai eps yang baik biasanya berada di sekitar titik tersebut."
    )

    k_nn = st.slider("k (nearest neighbors)", 3, 10, 5)

    @st.cache_data
    def compute_k_distance(coords_rad, k):
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="haversine").fit(coords_rad)
        distances, _ = nbrs.kneighbors(coords_rad)
        return np.sort(distances[:, k - 1])[::-1] * 6371  # km

    k_dist = compute_k_distance(coords_rad, k_nn)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(k_dist)), k_dist, "b-", linewidth=2)
    ax.axhline(y=eps_km, color="red", linestyle="--", label=f"eps saat ini = {eps_km} km")
    ax.set_xlabel("Kelurahan (diurutkan)")
    ax.set_ylabel(f"Jarak ke-{k_nn} tetangga terdekat (km)")
    ax.set_title("K-Distance Plot", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ─── FUNGSI DBSCAN ────────────────────────────────────────────────────────────
def run_dbscan(coords_rad, eps_km, min_samples):
    eps_rad = eps_km / 6371.0
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm="ball_tree", metric="haversine")
    return db.fit_predict(coords_rad)

def cluster_metrics(coords_rad, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    noise_ratio = round(n_noise / len(labels) * 100, 1)
    if n_clusters > 1:
        sil = round(silhouette_score(coords_rad, labels, metric="haversine"), 4)
        dbi = round(davies_bouldin_score(coords_rad, labels), 4)
        ch = round(calinski_harabasz_score(coords_rad, labels), 2)
    else:
        sil = dbi = ch = -1
    return n_clusters, n_noise, noise_ratio, sil, dbi, ch

# ─── TAB 3: EVALUASI PARAMETER ────────────────────────────────────────────────
with tab3:
    st.subheader("Evaluasi Komprehensif Parameter DBSCAN")

    if run_grid:
        eps_values = [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
        ms_values = [2, 3, 4, 5]

        @st.cache_data
        def grid_search(coords_rad, eps_values, ms_values):
            rows = []
            for eps in eps_values:
                for ms in ms_values:
                    labels = run_dbscan(coords_rad, eps, ms)
                    n_cl, n_noise, noise_ratio, sil, dbi, ch = cluster_metrics(coords_rad, labels)
                    if n_cl > 1:
                        rows.append({
                            "eps_km": eps, "min_samples": ms,
                            "n_clusters": n_cl, "n_noise": n_noise,
                            "noise_ratio_%": noise_ratio,
                            "silhouette": sil, "davies_bouldin": dbi,
                            "calinski_harabasz": ch,
                        })
            rdf = pd.DataFrame(rows)
            rdf["rank_score"] = (
                rdf["silhouette"].rank(ascending=True)
                + rdf["davies_bouldin"].rank(ascending=False)
                + rdf["calinski_harabasz"].rank(ascending=True)
            )
            return rdf.sort_values("rank_score", ascending=False).reset_index(drop=True)

        with st.spinner("Menjalankan grid search…"):
            results_df = grid_search(coords_rad, eps_values, ms_values)

        st.dataframe(results_df.style.background_gradient(subset=["silhouette"], cmap="YlGn")
                                     .background_gradient(subset=["davies_bouldin"], cmap="RdYlGn_r")
                                     .format(precision=4), use_container_width=True)

        st.subheader("Heatmap Evaluasi")
        metrics_list = ["silhouette", "davies_bouldin", "calinski_harabasz", "rank_score"]
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        axes2 = axes2.flatten()
        for i, metric in enumerate(metrics_list):
            pivot = results_df.pivot(index="eps_km", columns="min_samples", values=metric)
            cmap = "RdYlGn" if metric == "davies_bouldin" else "YlGnBu"
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, ax=axes2[i], linewidths=0.5)
            direction = "(rendah=baik)" if metric == "davies_bouldin" else "(tinggi=baik)"
            axes2[i].set_title(f"{metric.replace('_', ' ').title()}\n{direction}", fontweight="bold")
        plt.suptitle("Evaluasi Komprehensif Parameter DBSCAN", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # Rekomendasi otomatis
        best = results_df[results_df["noise_ratio_%"] <= 15].iloc[0]
        st.success(
            f"✅ **Parameter terbaik:** eps = {best['eps_km']} km, "
            f"min_samples = {int(best['min_samples'])} → "
            f"{int(best['n_clusters'])} klaster, noise {best['noise_ratio_%']}%, "
            f"silhouette {best['silhouette']}"
        )
    else:
        st.info("Centang **Jalankan Grid Search Parameter** di sidebar untuk menampilkan evaluasi lengkap.")

    # Preview klaster parameter saat ini
    st.divider()
    st.subheader(f"Preview Klaster (eps={eps_km} km, min_samples={min_samples})")
    labels_now = run_dbscan(coords_rad, eps_km, min_samples)
    n_cl, n_noise, noise_ratio, sil, dbi, ch = cluster_metrics(coords_rad, labels_now)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Klaster", n_cl)
    c2.metric("Noise", f"{n_noise} ({noise_ratio}%)")
    c3.metric("Silhouette", sil)
    c4.metric("Davies-Bouldin", dbi)

    kel_plot = kel.copy()
    kel_plot["cluster"] = labels_now
    palette = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
               "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4","#8bc34a"]

    fig3, ax3 = plt.subplots(figsize=(9, 6))
    unique_cl = sorted(kel_plot["cluster"].unique())
    for c in unique_cl:
        sub = kel_plot[kel_plot["cluster"] == c]
        color = "#95a5a6" if c == -1 else palette[c % len(palette)]
        label = "Noise" if c == -1 else f"Klaster {c}"
        marker = "x" if c == -1 else "o"
        ax3.scatter(sub["longtitude"], sub["latitude"],
                    c=color, s=sub["total_bank_sampah"] * 1.5,
                    alpha=0.75, edgecolors="white", linewidths=0.5,
                    label=label, marker=marker)
    ax3.set_title(f"Scatter Klaster | eps={eps_km} km, min_samples={min_samples}", fontweight="bold")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax3.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

# ─── CLUSTER LABELING ─────────────────────────────────────────────────────────
labels_final = run_dbscan(coords_rad, eps_km, min_samples)
kel_final = kel.copy()
kel_final["cluster"] = labels_final

cluster_avg = kel_final.groupby("cluster")["total_bank_sampah"].mean()

def label_cluster(row):
    if row["cluster"] == -1:
        return "Noise (Terisolasi)"
    avg = cluster_avg.get(row["cluster"], 0)
    if avg >= 40:
        return "Padat Tinggi"
    elif avg >= 20:
        return "Padat Sedang"
    return "Padat Rendah"

kel_final["label"] = kel_final.apply(label_cluster, axis=1)

# ─── TAB 4: PETA INTERAKTIF ────────────────────────────────────────────────────
with tab4:
    st.subheader("🗺️ Peta Interaktif Clustering Bank Sampah")

    palette_map = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
                   "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4","#8bc34a"]

    def get_color(c):
        return "#95a5a6" if c == -1 else palette_map[c % len(palette_map)]

    layer_choice = st.radio(
        "Tampilkan layer:",
        ["Klaster DBSCAN per Kelurahan", "Heatmap Kepadatan", "Heatmap Aktif", "Area Terisolasi (Noise)"],
        horizontal=True,
    )

    m = folium.Map(location=[-6.2, 106.816], zoom_start=11, tiles="CartoDB positron")

    if layer_choice == "Klaster DBSCAN per Kelurahan":
        for _, row in kel_final.iterrows():
            color = get_color(row["cluster"])
            radius = max(5, min(22, row["total_bank_sampah"] * 0.3))
            cluster_label = f"Klaster {row['cluster']}" if row["cluster"] != -1 else "Noise"
            popup_html = (
                f"<div style='font-family:Arial;min-width:190px'>"
                f"<b>{row['kelurahan']}</b><br>"
                f"<b>Wilayah:</b> {row['wilayah']}<br>"
                f"<b>Kecamatan:</b> {row['kecamatan']}<br>"
                f"<b>Total:</b> {int(row['total_bank_sampah'])} | "
                f"<span style='color:green'>Aktif: {int(row['jumlah_aktif'])}</span><br>"
                f"<b>% Aktif:</b> {row['pct_aktif']}%<br>"
                f"<b>Klaster:</b> <span style='background:{color};color:white;"
                f"padding:1px 5px;border-radius:3px'>{cluster_label}</span><br>"
                f"<b>Label:</b> {row['label']}</div>"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longtitude"]],
                radius=radius, color="white", weight=1.5,
                fill=True, fill_color=color, fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{row['kelurahan']} | {cluster_label} | {int(row['total_bank_sampah'])} bs",
            ).add_to(m)

    elif layer_choice == "Heatmap Kepadatan":
        heat_data = [[r["latitude"], r["longtitude"], r["total_bank_sampah"]] for _, r in kel_final.iterrows()]
        HeatMap(heat_data, radius=25, blur=20, min_opacity=0.3).add_to(m)

    elif layer_choice == "Heatmap Aktif":
        heat_aktif = [[r["latitude"], r["longtitude"], r["jumlah_aktif"]]
                      for _, r in kel_final[kel_final["jumlah_aktif"] > 0].iterrows()]
        HeatMap(heat_aktif, radius=25, blur=20, min_opacity=0.3,
                gradient={"0.4": "blue", "0.65": "green", "1": "lime"}).add_to(m)

    else:  # Noise
        noise_rows = kel_final[kel_final["cluster"] == -1]
        for _, row in noise_rows.iterrows():
            folium.Marker(
                location=[row["latitude"], row["longtitude"]],
                icon=folium.Icon(color="gray", icon="exclamation-sign", prefix="glyphicon"),
                popup=folium.Popup(
                    f"<b>⚠️ TERISOLASI</b><br>{row['kelurahan']}<br>"
                    f"{row['kecamatan']}<br>Total: {int(row['total_bank_sampah'])}",
                    max_width=200,
                ),
                tooltip=f"⚠️ {row['kelurahan']} (Terisolasi)",
            ).add_to(m)

    st_folium(m, width=1100, height=600, returned_objects=[])

# ─── TAB 5: TABEL HASIL ───────────────────────────────────────────────────────
with tab5:
    st.subheader("📋 Tabel Hasil Clustering")

    output_cols = ["kelurahan", "kecamatan", "wilayah", "latitude", "longtitude",
                   "total_bank_sampah", "jumlah_aktif", "jumlah_tidak_aktif",
                   "pct_aktif", "cluster", "label"]
    display_df = kel_final[output_cols].copy()

    search_q = st.text_input("🔍 Cari kelurahan / wilayah / label…")
    if search_q:
        mask = display_df.apply(lambda col: col.astype(str).str.contains(search_q, case=False)).any(axis=1)
        display_df = display_df[mask]

    st.dataframe(
        display_df.style.background_gradient(subset=["total_bank_sampah"], cmap="YlGn")
                        .format({"pct_aktif": "{:.1f}%", "latitude": "{:.6f}", "longtitude": "{:.6f}"}),
        use_container_width=True, height=450
    )

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV Hasil", data=csv_bytes,
                       file_name="hasil_klaster_bank_sampah.csv", mime="text/csv")

    st.divider()
    st.subheader("Ringkasan Klaster")
    n_cl_f, n_noise_f, noise_ratio_f, sil_f, dbi_f, ch_f = cluster_metrics(coords_rad, labels_final)

    summary = kel_final.groupby(["cluster", "label"]).agg(
        jumlah_kelurahan=("kelurahan", "count"),
        total_bank_sampah=("total_bank_sampah", "sum"),
        avg_bank=("total_bank_sampah", "mean"),
        avg_pct_aktif=("pct_aktif", "mean"),
    ).round(1).reset_index()
    st.dataframe(summary, use_container_width=True)

    st.info(
        f"**Kelurahan terisolasi (noise):** "
        f"{kel_final[kel_final['cluster']==-1][['kelurahan','total_bank_sampah','pct_aktif']].sort_values('total_bank_sampah').to_string(index=False)}"
    )
