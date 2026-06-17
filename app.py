import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Netflix Clustering", layout="centered", page_icon="🎬")

st.title("🎬 Netflix Shows Clustering System")
st.write("Aplikasi AI K-Means Clustering untuk pengelompokkan tayangan Netflix.")
st.write("Sumber Dataset: [Kaggle - Netflix](https://www.kaggle.com/datasets/shivamb/netflix-shows)")

@st.cache_resource
def load_model():
    with open('netflix_kmeans_model.pkl', 'rb') as f:
        kmeans, mlb, ohe, df = pickle.load(f)
    return kmeans, mlb, ohe, df

try:
    kmeans, mlb, ohe, df = load_model()
    
    # Menggunakan selectbox (dropdown) agar dosen tinggal klik dan TIDAK AKAN SALAH KETIK
    titles = df['title'].values
    selected_title = st.selectbox("Pilih judul tayangan Netflix di bawah ini:", titles)
    
    if st.button("Analisis Kelompok AI"):
        show_data = df[df['title'] == selected_title].iloc[0]
        cluster_id = show_data['cluster']
        
        st.success(f"Hasil Analisis untuk: {selected_title}")
        st.info(f"💡 Tayangan ini masuk ke dalam **Klaster {cluster_id}**")
        
        # Tampilkan rekomendasi
        st.write("🎞️ **Tayangan Lain dalam Kelompok yang Sama:**")
        rekomendasi = df[df['cluster'] == cluster_id][df['title'] != selected_title]['title'].head(3).tolist()
        for r in rekomendasi:
            st.write(f"- {r}")
except Exception as e:
    st.error(f"Terjadi kesalahan sistem: {e}")