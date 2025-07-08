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

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Analisis Sentimen Pawon", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: white !important;
    }
    .main {
        background-color: white !important;
    }
    .main-title {font-size:42px; font-weight:bold; color:#4CAF50; text-align:center;}
    .sub-title {font-size:20px; color:#666; text-align:center; margin-bottom:2em;}
    .highlight {background:#f0f7ff; border-radius:5px; padding:15px; margin:15px 0;}
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">📊 Analisis Sentimen Pawon Mbah Gito</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Menggunakan Metode Naive Bayes</div>', unsafe_allow_html=True)

# -------------------- LOAD & CLEAN --------------------
try:
    df = pd.read_csv("dataset_pawon.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("dataset_pawon.csv", encoding="ISO-8859-1")
except FileNotFoundError:
    st.error("❌ File 'dataset_pawon.csv' tidak ditemukan.")
    st.stop()

df.columns = ["Ulasan", "Rating"]
df = df.dropna(subset=["Ulasan"])
df = df[df["Ulasan"].str.strip() != ""]

# -------------------- PREPROCESSING --------------------
stemmer = StemmerFactory().create_stemmer()
stopwords = set([
    'yang','dan','di','ke','dari','untuk','dengan','pada','dalam','karena',
    'atau','seperti','jadi','agar','supaya','walaupun','meskipun','namun',
    'tetapi','bahkan','maupun','hingga','antara','tanpa','selama','sejak',
    'terhadap','oleh','saya','aku','kami','kita','anda','kamu','mereka',
    'nya','ia','saja','pun','sih','dong','nih','wah','apa','siapa','dimana',
    'kapan','mengapa','bagaimana','juga','lagi','adalah','itu','ini','ya',
    'kalau','semua','setiap','hanya','sudah','telah','pernah','sedang',
    'tersebut','lalu','dll','tsb'
])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stopwords]
    return stemmer.stem(" ".join(words))

@st.cache_data(show_spinner="🔄 Sedang preprocessing...")
def preprocess_all(series):
    return [preprocess_text(t) for t in series]

df["cleaned_text"] = preprocess_all(df["Ulasan"].astype(str))

def convert_rating_to_sentiment(r):
    try:
        r = int(r)
        if r in [1, 2]:   return 0
        elif r == 3:      return 1
        elif r in [4, 5]: return 2
    except:
        return None

df["sentiment"] = df["Rating"].apply(convert_rating_to_sentiment)
df = df.dropna(subset=["sentiment"])
df["sentiment"] = df["sentiment"].astype(int)

sentiment_labels = {0: "Negatif", 1: "Netral", 2: "Positif"}
df["sentimen_label"] = df["sentiment"].map(sentiment_labels)

# -------------------- OVERSAMPLING AT DF LEVEL --------------------
ros = RandomOverSampler(random_state=42)
X_df = df[["Ulasan", "cleaned_text"]]
y_df = df["sentiment"]

X_res_df, y_res = ros.fit_resample(X_df, y_df)
raw_resampled  = X_res_df["Ulasan"].values
texts_resampled = X_res_df["cleaned_text"].values
y_resampled    = y_res.values

df_oversampled = pd.DataFrame({
    "Ulasan": raw_resampled,
    "cleaned_text": texts_resampled,
    "sentiment": y_resampled
})
df_oversampled["sentimen_label"] = df_oversampled["sentiment"].map(sentiment_labels)

# -------------------- TF-IDF & TRAIN-TEST SPLIT --------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_resampled = vectorizer.fit_transform(texts_resampled).toarray()

# split raw and cleaned texts in sync with features
X_train, X_test, y_train, y_test, raw_train, raw_test, text_train, text_test = train_test_split(
    X_resampled, y_resampled,
    raw_resampled, texts_resampled,
    test_size=0.2, random_state=42, stratify=y_resampled
)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)

acc         = accuracy_score(y_test, y_pred)
report      = classification_report(y_test, y_pred,
                                    target_names=["Negatif","Netral","Positif"],
                                    output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# prepare results DataFrame
df_result = pd.DataFrame({
    "Teks Asli": raw_test,
    "Teks Bersih": text_test,
    "Aktual": y_test,
    "Prediksi": y_pred,
    "Confidence": y_proba.max(axis=1)
})
df_result["Aktual_Label"]   = df_result["Aktual"].map(sentiment_labels)
df_result["Prediksi_Label"] = df_result["Prediksi"].map(sentiment_labels)

# -------------------- BUILD UI WITH TABS --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Data", "📈 Evaluasi", "☁️ Word Cloud",
    "🔍 Prediksi Manual", "🧠 Fitur Penting"
])

# TAB 1: Data
with tab1:
    st.subheader("📄 Data & Preprocessing")
    with st.expander("ℹ️ Info Dataset"):
    st.markdown(f"""
    <div class="">
    <b>Dataset Pawon Mbah Gito</b><br>
    - <b>Data</b>: {len(df)}<br>
    - <b>Ulasan</b>: Teks ulasan pelanggan<br>
    - <b>Rating</b>: Nilai rating 1–5<br>
    - <b>Sentimen</b>: Kategori sentimen (Negatif, Netral, Positif)
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Contoh Data Asli**")
        st.dataframe(df[["Ulasan","Rating"]].head(5), use_container_width=True)
    with c2:
        st.markdown("**Contoh Cleaned**")
        st.dataframe(df[["cleaned_text","sentimen_label"]].head(5), use_container_width=True)

    st.subheader("Distribusi Sentimen")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.pie(df, names="sentimen_label",
                      title="Sebelum Oversampling", hole=0.4,
                      color="sentimen_label",
                      color_discrete_map={
                          "Negatif":"#EF553B",
                          "Netral":"#636EFA",
                          "Positif":"#00CC96"
                      })
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.pie(df_oversampled, names="sentimen_label",
                      title="Sesudah Oversampling", hole=0.4,
                      color="sentimen_label",
                      color_discrete_map={
                          "Negatif":"#EF553B",
                          "Netral":"#636EFA",
                          "Positif":"#00CC96"
                      })
        st.plotly_chart(fig2, use_container_width=True)

# TAB 2: Evaluasi
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.metric("🎯 Akurasi", f"{acc:.2%}")
        st.markdown("**📋 Classification Report**")
        rpt_df = pd.DataFrame(report).transpose()
        st.dataframe(rpt_df.style.format({
            "precision":"{:.2f}",
            "recall":"{:.2f}",
            "f1-score":"{:.2f}",
            "support":"{:.0f}"
        }), use_container_width=True)
    with c2:
        st.markdown("**🧩 Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negatif","Netral","Positif"],
                    yticklabels=["Negatif","Netral","Positif"])
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        st.pyplot(fig)

    st.markdown("### 🧠 Interpretasi")
    st.markdown(f"- Akurasi keseluruhan: **{acc:.2%}**")
    for lbl in ["Negatif","Netral","Positif"]:
        p = report[lbl]["precision"]; r = report[lbl]["recall"]; f1 = report[lbl]["f1-score"]
        st.markdown(f"  - **{lbl}**: precision {p:.2f}, recall {r:.2f}, f1 {f1:.2f}")

    st.subheader("🔍 Error Analysis")
    mism = df_result[df_result["Aktual"] != df_result["Prediksi"]]
    st.markdown(f"Total kesalahan: **{len(mism)}** dari **{len(df_result)}**")
    if not mism.empty:
        st.dataframe(mism[["Teks Asli","Aktual_Label","Prediksi_Label"]].head(5),
                     use_container_width=True)
    else:
        st.success("Semua prediksi benar!")

# TAB 3: Word Cloud
with tab3:
    st.subheader("☁️ Word Cloud per Sentimen")
    cols = st.columns(3)
    cmap = {"Negatif":"Reds","Netral":"Blues","Positif":"Greens"}
    for i, lbl in enumerate([0,1,2]):
        with cols[i]:
            name = sentiment_labels[lbl]
            texts = df[df["sentiment"] == lbl]["cleaned_text"]
            if texts.empty:
                st.warning(f"Tidak ada data untuk {name}")
            else:
                wc = WordCloud(width=400, height=300,
                               background_color="white",
                               colormap=cmap[name],
                               max_words=50
                              ).generate(" ".join(texts))
                fig, ax = plt.subplots(figsize=(5,4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

# TAB 4: Manual Prediction
with tab4:
    st.subheader("🔍 Prediksi Manual")
    user_input = st.text_area("Masukkan ulasan:", height=150)
    if st.button("Prediksi"):
        if not user_input.strip():
            st.warning("Mohon masukkan teks.")
        else:
            cleaned = preprocess_text(user_input)
            vec     = vectorizer.transform([cleaned]).toarray()
            pred    = model.predict(vec)[0]
            proba   = model.predict_proba(vec)[0]
            c1, c2 = st.columns([1,2])
            with c1:
                st.markdown("**Hasil Preprocessing**")
                st.code(cleaned, language="text")
                st.success(f"Prediksi: **{sentiment_labels[pred]}**")
                st.metric("Confidence", f"{proba.max()*100:.1f}%")
            with c2:
                prob_df = pd.DataFrame({
                    "Sentimen": ["Negatif","Netral","Positif"],
                    "Probabilitas": proba
                })
                fig = px.bar(prob_df, x="Sentimen", y="Probabilitas",
                             color="Sentimen",
                             color_discrete_map={
                                 "Negatif":"#EF553B",
                                 "Netral":"#636EFA",
                                 "Positif":"#00CC96"
                             },
                             text="Probabilitas", height=300)
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig.update_layout(yaxis_title="Probabilitas", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# TAB 5: Feature Importance
with tab5:
    st.subheader("🧠 Kata Paling Berpengaruh")
    features = vectorizer.get_feature_names_out()
    cols = st.columns(3)
    colors = {0:"#EF553B", 1:"#636EFA", 2:"#00CC96"}
    for i, lbl in enumerate([0,1,2]):
        with cols[i]:
            log_probs = model.feature_log_prob_[lbl]
            topn = 10
            idxs = np.argsort(log_probs)[-topn:][::-1]
            top_feats = [(features[j], np.exp(log_probs[j])) for j in idxs]
            df_top = pd.DataFrame(top_feats, columns=["Kata","Skor"])
            fig = px.bar(df_top, x="Skor", y="Kata", orientation="h",
                         title=f"Top {topn} {sentiment_labels[lbl]}",
                         color_discrete_sequence=[colors[lbl]], height=350)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔎 Analisis Kata Kunci")
    keyword = st.text_input("Cari kata kunci:")
    if keyword:
        kw = preprocess_text(keyword)
        try:
            idx = list(features).index(kw)
            scores = [np.exp(model.feature_log_prob_[i][idx]) for i in range(3)]
            df_kw = pd.DataFrame({
                "Sentimen": ["Negatif","Netral","Positif"],
                "Skor": scores
            })
            fig = px.bar(df_kw, x="Sentimen", y="Skor",
                         color="Sentimen",
                         color_discrete_map={
                             "Negatif":"#EF553B",
                             "Netral":"#636EFA",
                             "Positif":"#00CC96"
                         }, height=350)
            st.plotly_chart(fig, use_container_width=True)
            best = np.argmax(scores)
            st.markdown(f"""
                <div class="highlight">
                Kata **'{kw}'** paling memengaruhi **{sentiment_labels[best]}**
                (skor: {scores[best]:.4f})
                </div>
            """, unsafe_allow_html=True)
        except ValueError:
            st.warning(f"Kata '{kw}' tidak ditemukan.")
