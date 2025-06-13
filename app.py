import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Diagnosa FPV - ML", layout="centered")
st.title("🧪 Diagnosa FPV pada Kucing (ML-Based)")

# Load model dan encoder
model = joblib.load("fpv_model.pkl")
le = joblib.load("label_encoder.pkl")

# Pertanyaan input (25 indikator gejala)
def gejala_section(title, questions):
    st.subheader(title)
    result = []
    for q in questions:
        res = st.checkbox(q)
        result.append(1 if res else 0)
    return result

input_vector = []

input_vector += gejala_section("Diare", [
    "Apakah feses kucing terlihat sangat encer atau berlendir?",
    "Apakah warna feses kucing tampak kemerahan atau cokelat gelap (mungkin mengandung darah)?",
    "Apakah kucing buang air besar lebih sering dari biasanya (lebih dari 3x sehari)?",
    "Apakah kucing terlihat kesakitan atau mengejan saat buang air besar?",
    "Apakah bau feses kucing sangat menyengat dan tidak seperti biasanya?"
])

input_vector += gejala_section("Muntah Hebat", [
    "Apakah kucing muntah lebih dari 3 kali dalam sehari?",
    "Apakah muntahan mengandung makanan yang belum dicerna atau cairan empedu (kuning/hijau)?",
    "Apakah kucing terlihat sangat mual atau menjilat bibir terus-menerus sebelum muntah?",
    "Apakah tubuh kucing tegang atau terlihat sangat tidak nyaman saat muntah?"
])

input_vector += gejala_section("Lesu Berat", [
    "Apakah kucing terlihat tidak aktif sama sekali sepanjang hari?",
    "Apakah kucing menolak bermain atau tidak tertarik pada interaksi?",
    "Apakah tatapan mata kucing kosong dan tidak merespons ketika dipanggil?",
    "Apakah kucing lebih banyak tidur di tempat yang tidak biasa dan sulit dibangunkan?"
])

input_vector += gejala_section("Demam", [
    "Apakah tubuh kucing terasa lebih panas dari biasanya saat disentuh?",
    "Apakah telinga dan pangkal paha kucing terasa hangat atau panas?",
    "Apakah kucing terengah-engah meski tidak melakukan aktivitas berat?"
])

input_vector += gejala_section("Dehidrasi", [
    "Jika kulit kucing dicubit, apakah kulit lambat kembali ke posisi semula?",
    "Apakah gusi kucing terlihat kering, pucat, atau lengket?",
    "Apakah mata kucing terlihat cekung dan tidak cerah?"
])

input_vector += gejala_section("Penurunan Nafsu Makan", [
    "Apakah kucing menolak makanan favoritnya?",
    "Apakah kucing hanya mencium makanan lalu tidak memakannya?",
    "Apakah kucing tidak makan sama sekali dalam 24 jam terakhir?"
])

input_vector += gejala_section("Nyeri Perut", [
    "Apakah kucing menggeram atau mendesis saat perutnya disentuh?",
    "Apakah kucing sering meringkuk dalam posisi yang melindungi perutnya?",
    "Apakah kucing tampak bungkuk dan sering berpindah posisi karena tidak nyaman?"
])

# Prediksi
if st.button("🔍 Diagnosa Sekarang"):
    X_input = np.array(input_vector).reshape(1, -1)
    pred = model.predict(X_input)[0]
    confidence = model.predict_proba(X_input).max()
    label = le.inverse_transform([pred])[0]

    st.subheader("📋 Hasil Diagnosa:")
    st.success(f"Tingkat Infeksi: {label}")
    st.info(f"Tingkat Keyakinan Model: {confidence:.2f} (maks = 1.0)")

    if confidence < 0.6:
        st.warning("⚠️ Model tidak yakin dengan hasil ini. Pertimbangkan masukan lebih akurat atau konsultasi langsung ke dokter hewan.")
