# -------------------- IMPORT LIBRARY --------------------
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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
st.markdown('<div class="sub-title">Menggunakan Metode Naive Bayes</div>', unsafe_allow_html=True)

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

# -------------------- SPLIT & TRAINING --------------------
indices = np.arange(len(X_resampled))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_resampled, y_resampled, indices, test_size=0.2, random_state=42, stratify=y_resampled
)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Negatif', 'Netral', 'Positif'], output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Buat DataFrame untuk hasil prediksi
df_result = pd.DataFrame({
    'Teks Asli': df.iloc[resampled_indices[idx_test]]['Ulasan'].values,
    'Teks Bersih': texts_resampled[idx_test],
    'Aktual': y_test,
    'Prediksi': y_pred
})
df_result['Aktual_Label'] = df_result['Aktual'].map(sentiment_labels)
df_result['Prediksi_Label'] = df_result['Prediksi'].map(sentiment_labels)

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Data", "📈 Evaluasi", "☁ Word Cloud", "🔍 Prediksi Manual", "🧠 Fitur Penting"])

# -------------------- TAB 1: DATA --------------------
with tab1:
    st.subheader("📄 Data Awal dan Hasil Preprocessing")
    
    with st.expander("ℹ Informasi Dataset"):
        st.markdown("""
        <div class="highlight">
        <b>Dataset Ulasan Restoran Pawon Mbah Gito</b><br>
        - <b>Data</b>: 2154<br>
        - <b>Ulasan</b>: Teks ulasan pelanggan<br>
        - <b>Rating</b>: Nilai rating 1-5<br>
        - <b>Sentimen</b>: Kategori sentimen (Negatif, Netral, Positif)
        </div>
        """, unsafe_allow_html=True)
    
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
    col1, col2 = st.columns([1, 1])

    with col1:
        st.metric(label="🎯 Akurasi Model", value=f"{acc:.2%}")
        
        st.markdown("📋 Classification Report:")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 
                                          'f1-score': '{:.2f}', 'support': '{:.0f}'}), 
                   use_container_width=True)
        
        st.markdown("""
        <div class="highlight">
        <b>Penjelasan:</b> Tabel ini menunjukkan metrik evaluasi utama, yaitu precision, recall, dan f1-score untuk setiap kelas sentimen.<br>
        - <b>Precision</b>: Seberapa tepat prediksi model pada masing-masing kelas (minim false positive).<br>
        - <b>Recall</b>: Seberapa lengkap model mendeteksi data di kelas tersebut (minim false negative).<br>
        - <b>F1-score</b>: Gabungan precision dan recall.<br>
        Nilai yang tinggi menandakan performa yang baik.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("🧩 Confusion Matrix:")
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negatif', 'Netral', 'Positif'],
                    yticklabels=['Negatif', 'Netral', 'Positif'])
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        plt.title('Confusion Matrix')
        st.pyplot(fig_cm)
        
        st.markdown("""
        <div class="highlight">
        <b>Penjelasan:</b> Confusion Matrix menunjukkan jumlah prediksi yang benar (diagonal) dan salah (di luar diagonal).<br>
        Jika kotak diagonal besar, artinya model berhasil memprediksi dengan baik pada kelas tersebut.<br>
        Nilai di luar diagonal menunjukkan kesalahan klasifikasi antar kelas.
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("ℹ Informasi Model"):
        st.markdown("""
        - *Algoritma*: Multinomial Naive Bayes
        - *Vectorizer*: TF-IDF dengan n-gram (1,2)
        - *Jumlah Fitur*: 5,000
        - *Oversampling*: RandomOverSampler
        - *Train-Test Split*: 80-20
        """)
    
    st.markdown("""
    ### 🧠 Interpretasi Hasil Evaluasi

    Model Naive Bayes menunjukkan performa yang *sangat baik* secara keseluruhan dengan *akurasi sebesar {:.2f}%*.

    #### 📋 Classification Report:
    - *Negatif*: Precision {:.2f}, Recall {:.2f}, F1-score {:.2f}  
      → Model sangat baik dalam mengenali ulasan negatif, dengan recall hampir sempurna.
    - *Netral*: Precision {:.2f}, Recall {:.2f}, F1-score {:.2f}  
      → Model cukup akurat dalam menangkap ulasan netral, meskipun ada sedikit overlap dengan kelas lain.
    - *Positif*: Precision {:.2f}, Recall {:.2f}, F1-score {:.2f}  
      → Model sangat yakin saat memprediksi positif, tapi masih sering keliru mengklasifikasikan ulasan positif sebagai netral atau negatif.

    #### 🧩 Confusion Matrix:
    - Hampir semua ulasan *negatif* dan *netral* diprediksi dengan benar.
    - Namun, *ulasan positif* sering dikira netral atau bahkan negatif.
    - Ini menunjukkan bahwa model masih kesulitan membedakan ekspresi positif yang halus dari netral.

    """.format(
        acc*100,
        report['Negatif']['precision'], report['Negatif']['recall'], report['Negatif']['f1-score'],
        report['Netral']['precision'], report['Netral']['recall'], report['Netral']['f1-score'],
        report['Positif']['precision'], report['Positif']['recall'], report['Positif']['f1-score']
    ))

    st.subheader("🔍 Analisis Kesalahan Prediksi")
    mismatch = df_result[df_result['Aktual'] != df_result['Prediksi']]
    
    if not mismatch.empty:
        st.markdown(f"❌ Total Kesalahan:** {len(mismatch)} dari {len(df_result)} data ({len(mismatch)/len(df_result):.2%})")
        
        # Pilih sample acak
        n_show = min(5, len(mismatch))
        sampled_mismatch = mismatch.sample(n=n_show, random_state=None).copy()
        
        st.dataframe(sampled_mismatch[['Teks Asli', 'Aktual_Label', 'Prediksi_Label']], use_container_width=True)
        
        st.markdown("""
        <div class="highlight">
        <b>🔍 Kesimpulan:</b>
        - Model sangat baik dalam mengenali ulasan negatif dan netral
        - Perlu ditingkatkan dalam membedakan ulasan positif yang tidak eksplisit
        - Preprocessing tambahan seperti lemmatization atau penyesuaian stopwords bisa membantu
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ Tidak ada kesalahan prediksi!")
    
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
                    prediction = model.predict(vectorized)[0]
                    probs = model.predict_proba(vectorized)[0]
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"🧹 Hasil Preprocessing:")
                        st.code(cleaned, language='text')
                        
                        st.success(f"✅ Prediksi Sentimen: *{sentiment_labels[prediction]}*")
                        st.metric("Confidence Score", f"{max(probs)*100:.1f}%")
                    
                    with col2:
                        st.markdown("📊 Distribusi Probabilitas:")
                        prob_df = pd.DataFrame({
                            'Sentimen': ['Negatif', 'Netral', 'Positif'],
                            'Probabilitas': probs
                        })
                        
                        fig_prob = px.bar(
                            prob_df, 
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
                        fig_prob.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                        fig_prob.update_layout(
                            yaxis_title='Probabilitas',
                            xaxis_title='',
                            showlegend=False
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)
                        
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
            class_prob = model.feature_log_prob_[i]
            
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
            class_probs = [np.exp(model.feature_log_prob_[i][idx]) for i in range(3)]
            
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
            st.warning(f"Kata '{keyword_clean}' tidak ditemukan dalam model")
