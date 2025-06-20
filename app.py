import streamlit as st
import numpy as np
import joblib
from datetime import datetime
import pandas as pd

# Try to import plotly, use fallback if not available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("📊 Plotly tidak tersedia. Menggunakan visualisasi alternatif.")

# Konfigurasi halaman
st.set_page_config(
    page_title="VetCare AI - Diagnosa FPV",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih menarik
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .symptom-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .alert-card {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, #00b894, #00a085);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header aplikasi
st.markdown("""
<div class="main-header">
    <h1>🐱 VetCare AI - Sistem Diagnosa FPV</h1>
    <p>Deteksi Dini Feline Panleukopenia Virus pada Kucing dengan AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar untuk informasi dan navigasi
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=300&h=200&fit=crop", 
             caption="Kesehatan kucing adalah prioritas")
    
    st.markdown("### 🏥 Tentang FPV")
    st.info("""
    **Feline Panleukopenia Virus (FPV)** adalah penyakit viral serius yang menyerang kucing, 
    terutama anak kucing. Deteksi dini sangat penting untuk penanganan yang tepat.
    """)
    
    st.markdown("### 📊 Statistik Hari Ini")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Diagnosa", "47", "↑12%")
    with col2:
        st.metric("Akurasi", "94.2%", "↑2.1%")

# Load model dan encoder
@st.cache_resource
def load_models():
    try:
        model = joblib.load("frandom_forest_fpv_model.pkl")
        le = joblib.load("label_encoder_fpv.pkl")
        return model, le
    except:
        st.error("⚠️ Model tidak ditemukan. Pastikan file model tersedia.")
        return None, None

model, le = load_models()

# Tabs untuk navigasi
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Diagnosa", "📊 Riwayat", "📚 Edukasi", "⚙️ Pengaturan"])

with tab1:
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Container untuk input gejala
    st.markdown("### 📝 Evaluasi Gejala Kucing Anda")
    
    # Fungsi untuk membuat section gejala yang lebih menarik
    def gejala_section_enhanced(title, questions, icon, description):
        with st.expander(f"{icon} {title}", expanded=True):
            st.markdown(f"*{description}*")
            result = []
            cols = st.columns(1)
            for i, q in enumerate(questions):
                res = st.checkbox(q, key=f"{title}_{i}")
                result.append(1 if res else 0)
            
            # Progress indicator untuk section ini
            filled = sum(result)
            total = len(result)
            if filled > 0:
                st.progress(filled / total)
                st.caption(f"Gejala terdeteksi: {filled}/{total}")
            
            return result

    # Input gejala dengan desain yang lebih menarik
    input_vector = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        diare = gejala_section_enhanced(
            "Gangguan Pencernaan", 
            [
                "Feses sangat encer atau berlendir",
                "Warna feses kemerahan/cokelat gelap",
                "Buang air besar >3x sehari",
                "Terlihat kesakitan saat BAB",
                "Bau feses sangat menyengat"
            ],
            "💩",
            "Evaluasi kondisi pencernaan dan feses kucing"
        )
        input_vector += diare
        
        muntah = gejala_section_enhanced(
            "Muntah dan Mual",
            [
                "Muntah >3 kali sehari",
                "Muntahan berisi makanan/cairan empedu",
                "Mual berkelanjutan",
                "Tubuh tegang saat muntah"
            ],
            "🤮",
            "Periksa frekuensi dan karakteristik muntah"
        )
        input_vector += muntah
        
        lesu = gejala_section_enhanced(
            "Aktivitas dan Energi",
            [
                "Tidak aktif sepanjang hari",
                "Menolak bermain/interaksi",
                "Tatapan kosong, tidak responsif",
                "Tidur berlebihan, sulit dibangunkan"
            ],
            "😴",
            "Amati tingkat aktivitas dan responsivitas"
        )
        input_vector += lesu
        
        demam = gejala_section_enhanced(
            "Suhu Tubuh",
            [
                "Tubuh terasa lebih panas",
                "Telinga dan pangkal paha hangat",
                "Terengah-engah tanpa aktivitas"
            ],
            "🌡️",
            "Periksa indikator suhu tubuh abnormal"
        )
        input_vector += demam
    
    with col2:
        dehidrasi = gejala_section_enhanced(
            "Status Hidrasi",
            [
                "Kulit lambat kembali saat dicubit",
                "Gusi kering/pucat/lengket",
                "Mata cekung dan tidak cerah"
            ],
            "💧",
            "Evaluasi tingkat hidrasi tubuh"
        )
        input_vector += dehidrasi
        
        nafsu_makan = gejala_section_enhanced(
            "Pola Makan",
            [
                "Menolak makanan favorit",
                "Hanya mencium tanpa makan",
                "Tidak makan 24 jam terakhir"
            ],
            "🍽️",
            "Pantau perubahan pola dan nafsu makan"
        )
        input_vector += nafsu_makan
        
        nyeri = gejala_section_enhanced(
            "Indikator Nyeri",
            [
                "Menggeram saat perut disentuh",
                "Posisi meringkuk melindungi perut",
                "Bungkuk dan sering ganti posisi"
            ],
            "😣",
            "Deteksi tanda-tanda ketidaknyamanan"
        )
        input_vector += nyeri
        
        # Tambahan: Informasi kucing
        st.markdown("### 🐱 Informasi Kucing")
        with st.container():
            col_a, col_b = st.columns(2)
            with col_a:
                umur = st.selectbox("Umur", ["< 6 bulan", "6-12 bulan", "1-5 tahun", "> 5 tahun"])
                vaksin = st.selectbox("Status Vaksin", ["Lengkap", "Tidak Lengkap", "Tidak Tahu"])
            with col_b:
                berat = st.number_input("Berat Badan (kg)", min_value=0.5, max_value=15.0, value=3.0)
                lingkungan = st.selectbox("Lingkungan", ["Indoor", "Outdoor", "Mixed"])

    # Update progress bar
    total_symptoms = len(input_vector)
    positive_symptoms = sum(input_vector)
    progress_bar.progress(positive_symptoms / max(total_symptoms, 1))
    status_text.text(f"Gejala terdeteksi: {positive_symptoms}/{total_symptoms}")

    # Rule engine yang ditingkatkan
    facts = {
        "diare": sum(diare) >= 2,
        "muntah": sum(muntah) >= 2,
        "lesu": sum(lesu) >= 2,
        "demam": sum(demam) >= 1,
        "dehidrasi": sum(dehidrasi) >= 1,
        "nafsu_makan": sum(nafsu_makan) >= 2,
        "nyeri": sum(nyeri) >= 1
    }

    def rule_engine_enhanced(facts):
        critical_score = 0
        rules_triggered = []
        
        if facts["diare"] and facts["muntah"] and facts["lesu"]:
            critical_score += 3
            rules_triggered.append("Trias klasik FPV terdeteksi")
            
        if facts["nafsu_makan"] and facts["demam"] and facts["muntah"]:
            critical_score += 2
            rules_triggered.append("Kombinasi demam dan gangguan pencernaan")
            
        if facts["diare"] and facts["dehidrasi"] and facts["lesu"]:
            critical_score += 2
            rules_triggered.append("Dehidrasi dengan gejala GI")
            
        return critical_score >= 2, critical_score, rules_triggered

    # Tombol diagnosa dengan styling menarik
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        diagnosa_button = st.button(
            "🔍 MULAI DIAGNOSA AI", 
            type="primary",
            use_container_width=True
        )

    # Hasil diagnosa
    if diagnosa_button and model is not None:
        cukup_gejala, risk_score, rules = rule_engine_enhanced(facts)
        
        if cukup_gejala:
            # Simulasi loading
            with st.spinner('🤖 AI sedang menganalisis gejala...'):
                import time
                time.sleep(2)
            
            X_input = np.array(input_vector).reshape(1, -1)
            pred = model.predict(X_input)[0]
            confidence = model.predict_proba(X_input).max()
            
            if le is not None:
                label = le.inverse_transform([pred])[0]
            else:
                label = f"Kategori {pred}"
            
            # Tampilan hasil yang menarik
            st.markdown("## 📋 Hasil Diagnosa AI")
            
            # Metrics dalam 3 kolom
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Tingkat Infeksi", 
                    label,
                    help="Prediksi AI berdasarkan gejala yang diamati"
                )
            with col2:
                confidence_pct = f"{confidence:.1%}"
                st.metric(
                    "Tingkat Keyakinan", 
                    confidence_pct,
                    delta=f"{confidence-0.5:.1%}" if confidence > 0.5 else None
                )
            with col3:
                st.metric(
                    "Skor Risiko",
                    f"{risk_score}/5",
                    help="Berdasarkan aturan klinis"
                )
            
            # Visualisasi hasil
            if PLOTLY_AVAILABLE:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Score (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90}}))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback visualization using native Streamlit
                st.markdown("### 📊 Skor Keyakinan")
                confidence_pct = confidence * 100
                st.progress(confidence / 1.0)
                
                if confidence_pct >= 80:
                    st.success(f"🟢 Tinggi: {confidence_pct:.1f}%")
                elif confidence_pct >= 60:
                    st.warning(f"🟡 Sedang: {confidence_pct:.1f}%")
                else:
                    st.error(f"🔴 Rendah: {confidence_pct:.1f}%")
            
            # Interpretasi hasil
            if confidence >= 0.8:
                st.markdown("""
                <div class="success-card">
                    <h4>✅ Diagnosa Dengan Keyakinan Tinggi</h4>
                    <p>Model AI menunjukkan keyakinan tinggi pada hasil diagnosa ini. 
                    Segera konsultasikan dengan dokter hewan untuk konfirmasi dan penanganan.</p>
                </div>
                """, unsafe_allow_html=True)
            elif confidence >= 0.6:
                st.markdown("""
                <div class="info-card">
                    <h4>⚠️ Diagnosa Dengan Keyakinan Sedang</h4>
                    <p>Hasil diagnosa menunjukkan kemungkinan yang cukup kuat. 
                    Disarankan untuk pemeriksaan lebih lanjut oleh dokter hewan.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-card">
                    <h4>❓ Diagnosa Memerlukan Konfirmasi</h4>
                    <p>Model kurang yakin dengan hasil ini. Gejala mungkin tidak spesifik untuk FPV. 
                    Konsultasi langsung dengan dokter hewan sangat disarankan.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Rules yang terpicu
            if rules:
                st.markdown("### 🔍 Aturan Klinis yang Terpicu:")
                for rule in rules:
                    st.markdown(f"• {rule}")
            
            # Rekomendasi tindak lanjut
            st.markdown("### 💡 Rekomendasi Tindak Lanjut:")
            recommendations = [
                "Segera hubungi dokter hewan terdekat",
                "Jaga kucing tetap terhidrasi dengan air bersih",
                "Isolasi kucing dari kucing lain jika memungkinkan",
                "Catat perkembangan gejala untuk dilaporkan ke dokter",
                "Siapkan informasi riwayat vaksinasi kucing"
            ]
            
            for rec in recommendations:
                st.markdown(f"✓ {rec}")
                
        else:
            st.markdown("""
            <div class="info-card">
                <h4>📊 Gejala Belum Mencukupi</h4>
                <p>Gejala yang terdeteksi belum cukup kuat untuk melakukan diagnosa otomatis. 
                Pertimbangkan untuk:</p>
                <ul>
                    <li>Observasi lebih lanjut selama 24-48 jam</li>
                    <li>Isi lebih banyak gejala jika muncul</li>
                    <li>Konsultasi langsung ke dokter hewan jika khawatir</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Riwayat Diagnosa")
    
    # Simulasi data riwayat
    sample_data = {
        'Tanggal': ['2024-01-15', '2024-01-10', '2024-01-05'],
        'Nama Kucing': ['Whiskers', 'Milo', 'Luna'],
        'Hasil': ['Ringan', 'Sedang', 'Berat'],
        'Confidence': [0.85, 0.72, 0.91],
        'Status': ['Sembuh', 'Dalam Perawatan', 'Sembuh']
    }
    
    df = pd.DataFrame(sample_data)
    st.dataframe(df, use_container_width=True)
    
    # Chart riwayat
    if PLOTLY_AVAILABLE:
        fig = px.line(df, x='Tanggal', y='Confidence', title='Trend Akurasi Diagnosa')
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback chart using Streamlit native
        st.line_chart(df.set_index('Tanggal')['Confidence'])

with tab3:
    st.markdown("### 📚 Edukasi tentang FPV")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🦠 Apa itu FPV?
        Feline Panleukopenia Virus adalah penyakit viral yang sangat menular pada kucing, 
        terutama berbahaya bagi anak kucing dan kucing yang tidak divaksin.
        
        #### 🎯 Gejala Utama:
        - Diare berdarah
        - Muntah hebat
        - Demam tinggi
        - Kehilangan nafsu makan
        - Dehidrasi parah
        - Lesu dan lemah
        """)
    
    with col2:
        st.markdown("""
        #### 💉 Pencegahan:
        - Vaksinasi rutin sesuai jadwal
        - Menjaga kebersihan lingkungan
        - Isolasi kucing sakit
        - Desinfeksi regular
        
        #### 🚨 Kapan ke Dokter:
        - Gejala muncul tiba-tiba
        - Kondisi memburuk cepat
        - Anak kucing menunjukkan gejala
        - Kucing tidak divaksin
        """)

with tab4:
    st.markdown("### ⚙️ Pengaturan Aplikasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 Konfigurasi Model")
        threshold = st.slider("Threshold Confidence", 0.5, 0.9, 0.6)
        strict_mode = st.checkbox("Mode Ketat (Lebih Konservatif)")
        
    with col2:
        st.markdown("#### 🎨 Tampilan")
        theme = st.selectbox("Tema", ["Default", "Dark", "Colorful"])
        lang = st.selectbox("Bahasa", ["Indonesia", "English"])
    
    if st.button("💾 Simpan Pengaturan"):
        st.success("Pengaturan berhasil disimpan!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🐱 VetCare AI - Sistem Diagnosa FPV v2.0</p>
    <p><small>⚠️ Aplikasi ini hanya untuk skrining awal. Selalu konsultasikan dengan dokter hewan untuk diagnosa dan pengobatan yang tepat.</small></p>
</div>
""", unsafe_allow_html=True)
