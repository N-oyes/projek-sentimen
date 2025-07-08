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
from collections import Counter

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
    'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'dalam',
    'karena', 'atau', 'seperti', 'jadi', 'agar', 'supaya', 'walaupun', 'meskipun',
    'namun', 'tetapi', 'bahkan', 'maupun', 'hingga', 'antara', 'tanpa', 'selama',
    'sejak', 'terhadap', 'oleh', 'saya', 'aku', 'kami', 'kita', 'anda', 'kamu',
    'mereka', 'nya', 'ia', 'saja', 'pun', 'sih', 'dong', 'nih', 'wah',
    'apa', 'siapa', 'dimana', 'kapan', 'mengapa', 'bagaimana',
    'juga', 'lagi', 'adalah', 'itu', 'ini', 'ya', 'kalau', 'semua', 'setiap',
    'hanya', 'sudah', 'telah', 'pernah', 'sedang', 'tersebut', 'lalu', 'dll', 'tsb'
])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [word for word in text.split() if word not in stopwords]
    return stemmer.stem(' '.join(words))

@st.cache_data(show_spinner="🔄 Sedang melakukan preprocessing...")
def preprocess_all_texts(text_series):
    return [preprocess_text(text) for text in text_series]

def convert_rating_to_sentiment(rating):
    try:
        rating = int(rating)
        if rating in [1, 2]: return 0
        elif rating == 3: return 1
        elif rating in [4, 5]: return 2
        else: return None
    except:
        return None

df = df.dropna(subset=['Ulasan'])
df = df[df['Ulasan'].str.strip() != '']
df['cleaned_text'] = preprocess_all_texts(df['Ulasan'].astype(str))
df['sentiment'] = df['Rating'].apply(convert_rating_to_sentiment)
df = df.dropna(subset=['sentiment'])
df = df[df['cleaned_text'].str.strip() != '']

sentiment_labels = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
df['sentimen_label'] = df['sentiment'].map(sentiment_labels)

# -------------------- TF-IDF + OVERSAMPLING --------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['sentiment'].values
texts = df['cleaned_text'].values

# Check shapes and types
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("NaN values in X:", np.isnan(X).sum())
print("NaN values in y:", np.isnan(y).sum())
print("Data type of X:", X.dtype)
print("Data type of y:", y.dtype)

# Check if y has more than one unique class
if len(np.unique(y)) > 1:
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    resampled_indices = ros.sample_indices_
    texts_resampled = texts[resampled_indices]
else:
    # If only one class, no need for oversampling
    X_resampled, y_resampled, texts_resampled = X, y, texts
    resampled_indices = np.arange(len(y))

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

df_result = pd.DataFrame({
    'Teks Asli': df.iloc[resampled_indices[idx_test]]['Ulasan'].values,
    'Teks Bersih': texts_resampled[idx_test],
    'Aktual': y_test,
    'Prediksi': y_pred
})
df_result['Aktual_Label'] = df_result['Aktual'].map(sentiment_labels)
df_result['Prediksi_Label'] = df_result['Prediksi'].map(sentiment_labels)

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Data", "📈 Evaluasi", "☁️ Word Cloud", "🔍 Prediksi Manual", "🧠 Fitur Penting"])

# -------------------- TAB 1: DATA --------------------
with tab1:
    st.subheader("📄 Data Awal dan Hasil Preprocessing")
    st.dataframe(df[['Ulasan', 'Rating', 'cleaned_text', 'sentimen_label']].head(), use_container_width=True)

    st.subheader("📊 Distribusi Sentimen")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sebelum Oversampling**")
        fig1 = px.pie(df, names='sentimen_label', title='Distribusi Sentimen Awal')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown("**Setelah Oversampling**")
        fig2 = px.pie(df_oversampled, names='sentimen_label', title='Distribusi Setelah Oversampling')
        st.plotly_chart(fig2, use_container_width=True)

# -------------------- TAB 2: EVALUASI --------------------
with tab2:
    st.metric("Akurasi", f"{acc:.2%}")
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'])
    st.pyplot(fig_cm)

# -------------------- TAB 3: WORD CLOUD --------------------
with tab3:
    st.subheader("☁️ Word Cloud per Sentimen")
    cols = st.columns(3)
    for i, label in enumerate(['Negatif', 'Netral', 'Positif']):
        with cols[i]:
            text_data = df[df['sentimen_label'] == label]['cleaned_text']
            if not text_data.empty:
                text = ' '.join(text_data)
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

# -------------------- TAB 4: PREDIKSI MANUAL --------------------
with tab4:
    st.subheader("Prediksi Manual")
    user_input = st.text_area("Masukkan ulasan:")
    if st.button("Prediksi"):
        cleaned = preprocess_text(user_input)
        vectorized = vectorizer.transform([cleaned]).toarray()
        pred = model.predict(vectorized)[0]
        st.success(f"Prediksi Sentimen: **{sentiment_labels[pred]}**")

# -------------------- TAB 5: FITUR PENTING --------------------
with tab5:
    st.subheader("Fitur Penting")
    feature_names = vectorizer.get_feature_names_out()
    for label_num in [0, 1, 2]:
        st.markdown(f"**Top kata untuk {sentiment_labels[label_num]}**")
        class_prob = model.feature_log_prob_[label_num]
        top_indices = np.argsort(class_prob)[-10:][::-1]
        top_features = [(feature_names[i], np.exp(class_prob[i])) for i in top_indices]
        df_top = pd.DataFrame(top_features, columns=['Kata', 'Skor'])
        st.dataframe(df_top, use_container_width=True)
