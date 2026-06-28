import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dashboard Optimasi Bank Sampah", 
    page_icon="♻️", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        color: #266480;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        color: #555555;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">♻️ Smart Environment: Optimasi Bank Sampah DKI Jakarta</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pemetaan & Klasterisasi Wilayah Defisit Layanan menggunakan Algoritma DBSCAN</div>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def load_data():
    file_path = 'banksampah_clean_baru.csv'
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File {file_path} tidak ditemukan! Pastikan file berada di folder yang sama dengan app.py")
        return pd.DataFrame()

    clean_df = df.copy()
    if 'longtitude' in clean_df.columns:
        clean_df = clean_df.rename(columns={'longtitude': 'longitude'})
        
    clean_df = clean_df.dropna(subset=['latitude', 'longitude', 'nama_bank_sampah'])
    clean_df = clean_df.drop_duplicates()
    
    if 'status_kegiatan' in clean_df.columns:
        clean_df['status_kegiatan'] = clean_df['status_kegiatan'].fillna('TIDAK DIKETAHUI')
        
    return clean_df

data_bank_sampah = load_data()

if not data_bank_sampah.empty:
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3299/3299935.png", width=100)
        st.header("⚙️ Parameter DBSCAN")
        st.markdown("Geser *slider* di bawah untuk melihat perubahan klaster secara *real-time*.")
        
        eps_val = st.slider("Radius Pencarian (Epsilon)", min_value=0.10, max_value=1.00, value=0.40, step=0.05)
        min_samples_val = st.slider("Min Titik (Min Samples)", min_value=5, max_value=50, value=10, step=1)
        
        st.markdown("---")
        st.markdown("### 👥 Tim Newton (STT-NF)")
        st.markdown("""
        - **Rohmatul Hidayat** (01100224015)
        - **Anwar Maulana** (01100224020)
        - **M. Ridwan Karim** (01100224122)
        - **A. Muflih Alrasyid** (01100224162)
        """)
    coords = data_bank_sampah[['latitude', 'longitude']]
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)

    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    data_bank_sampah['Cluster'] = dbscan.fit_predict(scaled_coords)

    # Menghitung Metrik
    n_clusters = len(set(data_bank_sampah['Cluster'])) - (1 if -1 in data_bank_sampah['Cluster'].values else 0)
    n_noise = list(data_bank_sampah['Cluster']).count(-1)
    total_data = len(data_bank_sampah)
    
    aktif_count = len(data_bank_sampah[data_bank_sampah['status_kegiatan'].str.contains('AKTIF', na=False, case=False)])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bank Sampah", f"{total_data:,}".replace(',','.'))
    col2.metric("Bank Sampah Aktif", f"{aktif_count:,}".replace(',','.'))
    col3.metric("Klaster Padat Terbentuk", n_clusters)
    col4.metric("Titik Noise (Defisit Layanan)", n_noise, delta="Prioritas Perbaikan", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🗺️ Peta Interaktif DBSCAN", "📊 Analisis Grafik", "📋 Tabel Data Lengkap"])

    # --- PETA INTERAKTIF ---
    with tab1:
        st.markdown("#### Pemetaan Zona Layanan & Area Terisolasi (*Noise*)")
        
        map_center = [data_bank_sampah['latitude'].mean(), data_bank_sampah['longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=11.5, tiles='CartoDB positron')

        palette = sns.color_palette("tab10", n_clusters)
        colors_dict = {-1: "black"}
        for i in range(n_clusters):
            colors_dict[i] = '#%02x%02x%02x' % tuple(int(c * 255) for c in palette[i])

        for _, row in data_bank_sampah.iterrows():
            cluster_id = row['Cluster']
            is_noise = (cluster_id == -1)
            
            color = 'black' if is_noise else colors_dict.get(cluster_id, 'gray')
            radius_size = 6 if is_noise else 3.5
            status_color = "green" if "AKTIF" in str(row['status_kegiatan']).upper() else "red"
            cluster_text = f"<span style='color:red; font-weight:bold;'>⚠️ NOISE (TERISOLASI)</span>" if is_noise else f"<span style='color:blue; font-weight:bold;'>✅ Cluster {cluster_id}</span>"
            
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 250px;">
                <h4 style="color: #266480; margin-bottom: 5px;">{row['nama_bank_sampah']}</h4>
                <p style="margin: 0px; font-size: 12px; color: gray;">{row.get('alamat_lengkap', row.get('alamat', ''))}</p>
                <hr style="margin: 10px 0px;">
                <b>Wilayah:</b> {row['wilayah']}<br>
                <b>Status Operasional:</b> <span style="color: {status_color}; font-weight: bold;">{row['status_kegiatan']}</span><br>
                <b>Status DBSCAN:</b> {cluster_text}
            </div>
            """

            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius_size,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8 if is_noise else 0.4,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=row['nama_bank_sampah']
            ).add_to(m)

        components.html(m._repr_html_(), height=550)

    # --- ANALISIS GRAFIK ---
    with tab2:
        st.markdown("#### Distribusi Bank Sampah per Wilayah Administratif")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=data_bank_sampah, y='wilayah', order=data_bank_sampah['wilayah'].value_counts().index, palette='viridis', ax=ax)
        ax.set_title("Jumlah Bank Sampah di Setiap Kota Administrasi", fontweight='bold')
        ax.set_xlabel("Jumlah Fasilitas")
        ax.set_ylabel("Wilayah")
        st.pyplot(fig)
        
        st.info("💡 **Insight:** Grafik di atas menunjukkan perbandingan jumlah fasilitas bank sampah yang terdaftar di masing-masing kota administratif. Wilayah dengan jumlah fasilitas paling sedikit berisiko memiliki lebih banyak area *noise* (defisit layanan).")

    # --- TABEL DATA ---
    with tab3:
        st.markdown("#### Detail Dataset Hasil Klasterisasi")
        st.dataframe(
            data_bank_sampah[['nama_bank_sampah', 'wilayah', 'kecamatan', 'kelurahan', 'status_kegiatan', 'Cluster']],
            use_container_width=True,
            height=500
        )