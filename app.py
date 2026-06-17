import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Netflix Movie Recommendation", layout="centered", page_icon="🎬")

st.title("🎬 Netflix Movie Recommendation System")
st.write("Aplikasi Rekomendasi Film Berbasis **K-Means Clustering (Machine Learning)**.")
st.write("Sumber Dataset Resmi: [Kaggle - Netflix Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)")

# Fungsi memuat model pkl
@st.cache_resource
def load_model():
    with open('netflix_kmeans_model.pkl', 'rb') as f:
        kmeans, mlb, ohe, df = pickle.load(f)
    return kmeans, mlb, ohe, df

try:
    kmeans, mlb, ohe, df = load_model()
    
    # Filter khusus untuk tipe 'Movie' saja agar outputnya murni rekomendasi film
    df_movies = df[df['type'] == 'Movie']
    movie_titles = df_movies['title'].values
    
    # Dropdown pilihan judul film
    selected_movie = st.selectbox("Pilih judul film Netflix yang kamu sukai:", movie_titles)
    
    if st.button("Cari Rekomendasi Film"):
        # Ambil data film yang dipilih user
        movie_data = df_movies[df_movies['title'] == selected_movie].iloc[0]
        cluster_id = movie_data['cluster']
        
        st.success(f"Kamu memilih film: **{selected_movie}**")
        st.write(f"ℹ️ *Karakteristik Film Ini: Rating Usia ({movie_data['rating']}) | Genre ({movie_data['listed_in']})*")
        
        st.markdown("---")
        st.subheader("🍿 Rekomendasi Film Serupa untuk Kamu:")
        
        # Cari film lain yang berada di klaster kelompok yang sama
        recommendations = df_movies[
            (df_movies['cluster'] == cluster_id) & 
            (df_movies['title'] != selected_movie)
        ][['title', 'rating', 'listed_in']].header("Film Serupa").head(5) # Ambil 5 rekomendasi teratas
        
        # Ubah nama kolom tabel agar rapi saat dibaca dosen
        recommendations.columns = ['Judul Film', 'Rating Usia', 'Genre / Kategori']
        
        # Tampilkan dalam bentuk tabel interaktif
        st.dataframe(recommendations, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Terjadi kesalahan sistem: {e}")