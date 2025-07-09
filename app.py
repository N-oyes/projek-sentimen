# -------------------- IMPORT LIBRARY --------------------
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
from imblearn.over_sampling import RandomOverSampler
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# -------------------- KONFIGURASI HALAMAN --------------------
st.set_page_config(page_title="Analisis Sentimen Pawon", layout="wide")

st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .sub-title {
        font-size: 20px;
        color: #666;
        text-align: center;
        margin-bottom: 2em;
    }
    .highlight {
        background-color: #f0f7ff;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    @media (max-width: 600px) {
        .stPlotlyChart, .stDataFrame {
            width: 100% !important;
        }
        .column {
            min-width: 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Analisis Sentimen Pawon Mbah Gito</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Perbandingan Model Naive Bayes vs Random Forest</div>', unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
try:
    df = pd.read_csv("dataset_pawon.csv", encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv("dataset_pawon.csv", encoding='ISO-8859-1')
except FileNotFoundError:
    st.error("❌ File 'dataset_pawon.csv' tidak ditemukan.")
    st.stop()

df.columns = ['Ulasan', 'Rating']

# -------------------- PREPROCESSING --------------------
stemmer = StemmerFactory().create_stemmer()

stopwords = set([
    # Preposisi & konjungsi
    'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'dalam', 
    'karena', 'atau', 'seperti', 'jadi', 'agar', 'supaya', 'walaupun', 'meskipun', 
    'namun', 'tetapi', 'bahkan', 'maupun', 'hingga', 'antara', 'tanpa', 'selama', 
    'sejak', 'terhadap', 'oleh', 
    
    # Kata ganti & partikel
    'saya', 'aku', 'kami', 'kita', 'anda', 'kamu', 'mereka', 'nya', 'ia', 'saja', 
    'pun', 'sih', 'dong', 'nih', 'wah', 
    
    # Kata tanya
    'apa', 'siapa', 'dimana', 'kapan', 'mengapa', 'bagaimana', 
    
    # Adverbia & lainnya
    'juga', 'lagi', 'adalah', 'itu', 'ini', 'ya', 'kalau', 'semua', 'setiap', 
    'hanya', 'sudah', 'telah', 'pernah', 'sedang', 'tersebut', 'lalu', 'dll', 'tsb'
])

def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # hapus karakter non-alfabet
    text = re.sub(r'\s+', ' ', text).strip()  # hapus spasi berlebih
    words = [word for word in text.split() if word not in stopwords]  # hapus stopword
    return stemmer.stem(' '.join(words))  # stemming

def preprocess_all_texts(text_series):
    progress_bar = st.progress(0)
    results = []
    total = len(text_series)
    for i, text in enumerate(text_series):
        results.append(preprocess_text(text))
        progress_bar.progress((i + 1) / total)
    progress_bar.empty()
    return results

def convert_rating_to_sentiment(rating):
    try:
        rating = int(rating)
        if rating in [1, 2]: return 0
        elif rating == 3: return 1
        elif rating in [4, 5]: return 2
        else: return None  # Handle invalid ratings
    except:
        return None

# Bersihkan data
df = df.dropna(subset=['Ulasan'])
df = df[df['Ulasan'].str.strip() != '']
df['cleaned_text'] = preprocess_all_texts(df['Ulasan'].astype(str))
df['sentiment'] = df['Rating'].apply(convert_rating_to_sentiment)
df = df.dropna(subset=['sentiment'])  # Hapus baris dengan sentiment None
df = df[df['cleaned_text'].str.strip() != '']  # Hapus teks kosong

sentiment_labels = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
df['sentimen_label'] = df['sentiment'].map(sentiment_labels)

# -------------------- TF-IDF + OVERSAMPLING --------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['sentiment'].values
texts = df['cleaned_text'].values

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
resampled_indices = ros.sample_indices_
texts_resampled = texts[resampled_indices]

df_oversampled = pd.DataFrame({'sentiment': y_resampled})
df_oversampled['sentimen_label'] = df_oversampled['sentiment'].map(sentiment_labels)

# -------------------- SPLIT DATA --------------------
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_resampled, y_resampled, np.arange(len(X_resampled)), 
    test_size=0.2, random_state=42, stratify=y_resampled
)

# -------------------- TRAINING & EVALUASI --------------------
# Model Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb, target_names=['Negatif', 'Netral', 'Positif'], output_dict=True)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)

# Model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, target_names=['Negatif', 'Netral', 'Positif'], output_dict=True)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Buat DataFrame untuk hasil prediksi
df_result = pd.DataFrame({
    'Teks Asli': df.iloc[resampled_indices[idx_test]]['Ulasan'].values,
    'Teks Bersih': texts_resampled[idx_test],
    'Aktual': y_test,
    'Prediksi_NB': y_pred_nb,
    'Prediksi_RF': y_pred_rf
})
df_result['Aktual_Label'] = df_result['Aktual'].map(sentiment_labels)
df_result['Prediksi_NB_Label'] = df_result['Prediksi_NB'].map(sentiment_labels)
df_result['Prediksi_RF_Label'] = df_result['Prediksi_RF'].map(sentiment_labels)

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Data", "📈 Evaluasi", "☁ Word Cloud", "🔍 Prediksi Manual", "🧠 Fitur Penting"])

# -------------------- TAB 1: DATA --------------------
with tab1:
    st.subheader("📄 Data Awal dan Hasil Preprocessing")
    
    with st.expander("ℹ Informasi Dataset"):
        st.markdown("""
        <div class="highlight">
        <b>Dataset Ulasan Restoran Pawon Mbah Gito</b><br>
        - <b>Data</b>: {} baris<br>
        - <b>Ulasan</b>: Teks ulasan pelanggan<br>
        - <b>Rating</b>: Nilai rating 1-5<br>
        - <b>Sentimen</b>: Kategori sentimen (Negatif, Netral, Positif)
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    st.subheader("📊 Distribusi Rating Awal")
    fig_rating = px.histogram(df, x='Rating', nbins=5, 
                            title='Distribusi Rating 1-5',
                            color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_rating, use_container_width=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("📌 Contoh Data Asli:")
        st.dataframe(df[['Ulasan', 'Rating']].head(5), use_container_width=True)
        
    with col2:
        st.markdown("🧹 Contoh Hasil Preprocessing:")
        st.dataframe(df[['cleaned_text', 'sentimen_label']].head(5), use_container_width=True)
    
    st.subheader("📊 Distribusi Sentimen")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("*Sebelum Oversampling*")
        fig1 = px.pie(df, names='sentimen_label', title='Distribusi Sentimen Awal', 
                      hole=0.4, color='sentimen_label',
                      color_discrete_map={'Negatif': '#EF553B', 'Netral': '#636EFA', 'Positif': '#00CC96'})
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.markdown("*Setelah Oversampling*")
        fig2 = px.pie(df_oversampled, names='sentimen_label', title='Distribusi Setelah Oversampling', 
                      hole=0.4, color='sentimen_label',
                      color_discrete_map={'Negatif': '#EF553B', 'Netral': '#636EFA', 'Positif': '#00CC96'})
        st.plotly_chart(fig2, use_container_width=True)

# -------------------- TAB 2: EVALUASI --------------------
with tab2:
    st.subheader("Perbandingan Model")
    
    # Tampilkan metrik referensi
    st.markdown("### Metrik Referensi")
    reference_metrics = {
        'Metrik': ['Akurasi', 'Precision (Negatif)', 'Recall (Negatif)', 'F1-Score (Negatif)',
                   'Precision (Netral)', 'Recall (Netral)', 'F1-Score (Netral)',
                   'Precision (Positif)', 'Recall (Positif)', 'F1-Score (Positif)',
                   'Macro Avg F1', 'Weighted Avg F1'],
        'Naive Bayes': [0.91, 0.90, 0.90, 0.94, 
                        0.89, 0.89, 0.97, 
                        0.96, 0.96, 0.83, 
                        0.91, 0.91],
        'Random Forest': [0.99, 0.98, 1.00, 0.99,
                          1.00, 1.00, 1.00,
                          1.00, 0.98, 0.99,
                          0.99, 0.99]
    }

    df_reference = pd.DataFrame(reference_metrics)
    st.dataframe(df_reference.style.format({
        'Naive Bayes': '{:.2f}',
        'Random Forest': '{:.2f}'
    }), use_container_width=True)

    st.markdown("""
    <div class="highlight">
    **Interpretasi Referensi:**
    - Random Forest memiliki performa lebih baik di hampir semua metrik
    - Akurasi Random Forest (99%) lebih tinggi dari Naive Bayes (91%)
    - F1-Score Random Forest konsisten tinggi di semua kelas
    </div>
    """, unsafe_allow_html=True)
    
    # Hasil aktual dari model
    st.markdown("### Hasil Implementasi")
    
    # Ringkasan performa
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Akurasi Naive Bayes", f"{acc_nb:.2%}", 
                 delta=f"{acc_nb - 0.91:.2%}" if acc_nb > 0.91 else f"{0.91 - acc_nb:.2%} lebih rendah")
        st.metric("🎯 Akurasi Random Forest", f"{acc_rf:.2%}", 
                 delta=f"{acc_rf - 0.99:.2%}" if acc_rf > 0.99 else f"{0.99 - acc_rf:.2%} lebih rendah")
    
    # Grafik perbandingan akurasi
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Bar(
        x=['Naive Bayes', 'Random Forest'],
        y=[acc_nb, acc_rf],
        text=[f"{acc_nb:.2%}", f"{acc_rf:.2%}"],
        textposition='auto',
        marker_color=['#1f77b4', '#2ca02c'],
        name='Implementasi'
    ))
    fig_acc.add_trace(go.Scatter(
        x=['Naive Bayes', 'Random Forest'],
        y=[0.91, 0.99],
        mode='markers+text',
        text=['Referensi', 'Referensi'],
        textposition='top center',
        marker=dict(size=15, color='red'),
        name='Referensi'
    ))
    fig_acc.update_layout(
        title='Perbandingan Akurasi',
        yaxis=dict(range=[0.8, 1.05]),
        showlegend=True
    )
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # Tabel perbandingan metrik
    st.markdown("### Detail Metrik")
    
    # Buat dataframe untuk perbandingan
    metrics = [
        'Precision (Negatif)', 'Recall (Negatif)', 'F1-Score (Negatif)',
        'Precision (Netral)', 'Recall (Netral)', 'F1-Score (Netral)',
        'Precision (Positif)', 'Recall (Positif)', 'F1-Score (Positif)',
        'Macro Avg F1', 'Weighted Avg F1'
    ]
    
    nb_values = [
        report_nb['Negatif']['precision'], report_nb['Negatif']['recall'], report_nb['Negatif']['f1-score'],
        report_nb['Netral']['precision'], report_nb['Netral']['recall'], report_nb['Netral']['f1-score'],
        report_nb['Positif']['precision'], report_nb['Positif']['recall'], report_nb['Positif']['f1-score'],
        report_nb['macro avg']['f1-score'], report_nb['weighted avg']['f1-score']
    ]
    
    rf_values = [
        report_rf['Negatif']['precision'], report_rf['Negatif']['recall'], report_rf['Negatif']['f1-score'],
        report_rf['Netral']['precision'], report_rf['Netral']['recall'], report_rf['Netral']['f1-score'],
        report_rf['Positif']['precision'], report_rf['Positif']['recall'], report_rf['Positif']['f1-score'],
        report_rf['macro avg']['f1-score'], report_rf['weighted avg']['f1-score']
    ]
    
    df_comparison = pd.DataFrame({
        'Metrik': metrics,
        'Naive Bayes (Implementasi)': nb_values,
        'Random Forest (Implementasi)': rf_values,
        'Naive Bayes (Referensi)': [0.90, 0.90, 0.94, 0.89, 0.89, 0.97, 0.96, 0.96, 0.83, 0.91, 0.91],
        'Random Forest (Referensi)': [0.98, 1.00, 0.99, 1.00, 1.00, 1.00, 1.00, 0.98, 0.99, 0.99, 0.99]
    })
    
    st.dataframe(df_comparison.style.format({
        'Naive Bayes (Implementasi)': '{:.2f}',
        'Random Forest (Implementasi)': '{:.2f}',
        'Naive Bayes (Referensi)': '{:.2f}',
        'Random Forest (Referensi)': '{:.2f}'
    }), use_container_width=True)
    
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Naive Bayes**")
        fig_nb = px.imshow(
            conf_matrix_nb,
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
            x=['Negatif', 'Netral', 'Positif'],
            y=['Negatif', 'Netral', 'Positif'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig_nb.update_layout(title='Naive Bayes')
        st.plotly_chart(fig_nb, use_container_width=True)
    
    with col2:
        st.markdown("**Random Forest**")
        fig_rf = px.imshow(
            conf_matrix_rf,
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
            x=['Negatif', 'Netral', 'Positif'],
            y=['Negatif', 'Netral', 'Positif'],
            text_auto=True,
            color_continuous_scale='Greens'
        )
        fig_rf.update_layout(title='Random Forest')
        st.plotly_chart(fig_rf, use_container_width=True)
    
    # Analisis perbedaan
    st.markdown("### Analisis Perbedaan")
    st.markdown(f"""
    **🔍 Perbandingan Implementasi vs Referensi:**
    - **Akurasi Naive Bayes**: Referensi 91% vs Implementasi {acc_nb:.2%} ({'+' if acc_nb > 0.91 else ''}{acc_nb - 0.91:.2%})
    - **Akurasi Random Forest**: Referensi 99% vs Implementasi {acc_rf:.2%} ({'+' if acc_rf > 0.99 else ''}{acc_rf - 0.99:.2%})
    
    **📊 Distribusi Kesalahan:**
    - Naive Bayes: {conf_matrix_nb.sum() - np.trace(conf_matrix_nb)} kesalahan
    - Random Forest: {conf_matrix_rf.sum() - np.trace(conf_matrix_rf)} kesalahan
    
    **💡 Rekomendasi:**
    - Untuk akurasi tertinggi, gunakan Random Forest
    - Untuk kecepatan dan interpretabilitas, gunakan Naive Bayes
    
    **⚠ Catatan:**
    Perbedaan hasil implementasi dengan referensi bisa disebabkan oleh:
    1. Perbedaan dataset yang digunakan
    2. Variasi dalam teknik preprocessing
    3. Perbedaan parameter model
    4. Random seed yang berbeda
    """)
    
    st.download_button(
        label="📥 Download Hasil Prediksi",
        data=df_result.to_csv(index=False).encode('utf-8'),
        file_name='hasil_prediksi_sentimen.csv',
        mime='text/csv'
    )

# -------------------- TAB 3: WORD CLOUD --------------------
with tab3:
    st.subheader("☁ Word Cloud per Sentimen")
    col1, col2, col3 = st.columns(3)
    
    sentiment_colors = {
        'Negatif': 'Reds',
        'Netral': 'Blues',
        'Positif': 'Greens'
    }
    
    for label, name, col in zip([0, 1, 2], ['Negatif', 'Netral', 'Positif'], [col1, col2, col3]):
        with col:
            text_data = df[df['sentiment'] == label]['cleaned_text']
            if not text_data.empty:
                text = ' '.join(text_data)
                wordcloud = WordCloud(
                    width=400, 
                    height=300, 
                    background_color='white',
                    colormap=sentiment_colors[name],
                    max_words=50
                ).generate(text)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                ax.set_title(f"Kata Kunci {name}")
                st.pyplot(fig)
            else:
                st.warning(f"Tidak ada data untuk sentimen {name}")

# -------------------- TAB 4: PREDIKSI MANUAL --------------------
with tab4:
    st.subheader("🔍 Coba Prediksi Manual")
    
    with st.expander("💡 Tips Ulasan Contoh"):
        st.markdown("""
        - *Positif*: "Rasanya enak banget, pelayanan ramah dan harga terjangkau"
        - *Netral*: "Makanan disajikan dalam piring biasa, tidak terlalu besar atau kecil."
        - *Negatif*: "Kebersihannya kurang, bahkan saya melihat meja yang belum dibersihkan."
        """)
    
    user_input = st.text_area("Masukkan ulasan:", height=150)
    
    if st.button("Prediksi", type="primary"):
        if not user_input.strip():
            st.warning("⚠ Mohon masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner("🔄 Sedang memproses ulasan..."):
                try:
                    cleaned = preprocess_text(user_input)
                    vectorized = vectorizer.transform([cleaned]).toarray()
                    
                    # Prediksi untuk kedua model
                    prediction_nb = nb_model.predict(vectorized)[0]
                    probs_nb = nb_model.predict_proba(vectorized)[0]
                    
                    prediction_rf = rf_model.predict(vectorized)[0]
                    probs_rf = rf_model.predict_proba(vectorized)[0]
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### Naive Bayes")
                        st.markdown(f"🧹 Hasil Preprocessing:")
                        st.code(cleaned, language='text')
                        
                        st.success(f"✅ Prediksi Sentimen: *{sentiment_labels[prediction_nb]}*")
                        st.metric("Confidence Score", f"{max(probs_nb)*100:.1f}%")
                        
                        st.markdown("📊 Distribusi Probabilitas:")
                        prob_df_nb = pd.DataFrame({
                            'Sentimen': ['Negatif', 'Netral', 'Positif'],
                            'Probabilitas': probs_nb
                        })
                        
                        fig_prob_nb = px.bar(
                            prob_df_nb, 
                            x='Sentimen', 
                            y='Probabilitas',
                            color='Sentimen',
                            color_discrete_map={
                                'Negatif': '#EF553B', 
                                'Netral': '#636EFA', 
                                'Positif': '#00CC96'
                            },
                            text='Probabilitas',
                            height=300
                        )
                        fig_prob_nb.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                        fig_prob_nb.update_layout(
                            yaxis_title='Probabilitas',
                            xaxis_title='',
                            showlegend=False
                        )
                        st.plotly_chart(fig_prob_nb, use_container_width=True)
                    
                    with col2:
                        st.markdown("### Random Forest")
                        st.markdown(f"🧹 Hasil Preprocessing:")
                        st.code(cleaned, language='text')
                        
                        st.success(f"✅ Prediksi Sentimen: *{sentiment_labels[prediction_rf]}*")
                        st.metric("Confidence Score", f"{max(probs_rf)*100:.1f}%")
                        
                        st.markdown("📊 Distribusi Probabilitas:")
                        prob_df_rf = pd.DataFrame({
                            'Sentimen': ['Negatif', 'Netral', 'Positif'],
                            'Probabilitas': probs_rf
                        })
                        
                        fig_prob_rf = px.bar(
                            prob_df_rf, 
                            x='Sentimen', 
                            y='Probabilitas',
                            color='Sentimen',
                            color_discrete_map={
                                'Negatif': '#EF553B', 
                                'Netral': '#636EFA', 
                                'Positif': '#00CC96'
                            },
                            text='Probabilitas',
                            height=300
                        )
                        fig_prob_rf.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                        fig_prob_rf.update_layout(
                            yaxis_title='Probabilitas',
                            xaxis_title='',
                            showlegend=False
                        )
                        st.plotly_chart(fig_prob_rf, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat memproses: {e}")

# -------------------- TAB 5: FITUR PENTING --------------------
with tab5:
    st.subheader("🧠 Kata-Kata Paling Berpengaruh")
    
    # Ambil kata-kata paling penting dari model
    feature_names = vectorizer.get_feature_names_out()
    
    col1, col2, col3 = st.columns(3)
    sentiment_colors = {
        0: '#EF553B',  # Negatif
        1: '#636EFA',  # Netral
        2: '#00CC96'   # Positif
    }
    
    for i, label in enumerate([0, 1, 2]):
        with [col1, col2, col3][i]:
            # Dapatkan probabilitas log untuk kelas ini
            class_prob = nb_model.feature_log_prob_[i]
            
            # Ambil 10 fitur teratas
            topn = 10
            top_indices = np.argsort(class_prob)[-topn:][::-1]
            top_features = [(feature_names[j], np.exp(class_prob[j])) for j in top_indices]
            
            # Buat dataframe untuk visualisasi
            df_top = pd.DataFrame(top_features, columns=['Kata', 'Skor'])
            
            # Visualisasi
            fig = px.bar(
                df_top, 
                y='Kata', 
                x='Skor',
                orientation='h',
                title=f"Top {topn} Kata untuk {sentiment_labels[label]}",
                color_discrete_sequence=[sentiment_colors[label]],
                height=400
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🔎 Analisis Kata Kunci")
    keyword = st.text_input("Cari kata kunci spesifik:")
    
    if keyword:
        keyword_clean = preprocess_text(keyword)
        try:
            idx = list(feature_names).index(keyword_clean)
            
            # Dapatkan probabilitas untuk setiap kelas
            class_probs = [np.exp(nb_model.feature_log_prob_[i][idx]) for i in range(3)]
            
            # Visualisasi
            fig = px.bar(
                x=['Negatif', 'Netral', 'Positif'],
                y=class_probs,
                color=['Negatif', 'Netral', 'Positif'],
                color_discrete_map={
                    'Negatif': '#EF553B', 
                    'Netral': '#636EFA', 
                    'Positif': '#00CC96'
                },
                labels={'x': 'Sentimen', 'y': 'Skor Pengaruh'},
                title=f"Pengaruh Kata: '{keyword_clean}'",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tampilkan interpretasi
            max_sentiment = np.argmax(class_probs)
            st.markdown(f"""
            <div class="highlight">
            <b>Interpretasi:</b> Kata <b>'{keyword_clean}'</b> memiliki pengaruh terkuat terhadap sentimen 
            <b>{sentiment_labels[max_sentiment]}</b> dengan skor <b>{class_probs[max_sentiment]:.4f}</b>
            </div>
            """, unsafe_allow_html=True)
            
        except ValueError:
            st.warning(f"Kata '{keyword_clean}' tidak ditemukan dalam model")
