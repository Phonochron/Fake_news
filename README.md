# Fake News Detection AI

Project ini adalah aplikasi deteksi berita palsu berbasis Machine Learning dan Deep Learning. Aplikasi dibuat dengan Streamlit dan menggunakan beberapa model klasifikasi teks untuk memprediksi apakah sebuah berita termasuk fake news atau real news.

## Fitur

- Input teks berita melalui web app Streamlit
- Preprocessing teks menggunakan NLTK
- Ekstraksi fitur menggunakan TF-IDF
- Prediksi dengan beberapa model:
  - Naive Bayes
  - Logistic Regression
  - Linear SVM
  - Voting Ensemble
  - LSTM Neural Network
- Menampilkan metrik evaluasi model
- Menampilkan confusion matrix
- Menampilkan daftar kata yang sering muncul pada berita palsu

## Struktur Project

```text
fake_news_streamlit_project/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   ├── Fake.csv
│   └── True.csv
├── models/
│   ├── ensemble.pkl
│   ├── logistic_regression.pkl
│   ├── lstm_model.h5
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   ├── tfidf_vectorizer.pkl
│   └── tokenizer.pkl
└── outputs/
    ├── confusion_matrix.png
    ├── metrics.csv
    └── top_hoax_words.csv
```

## Instalasi

1. Clone repository:

```bash
git clone https://github.com/USERNAME/NAMA_REPO.git
cd NAMA_REPO
```

2. Buat virtual environment:

```bash
python -m venv .venv
```

3. Aktifkan virtual environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

4. Install dependency:

```bash
pip install -r requirements.txt
```

## Menjalankan Training

Pastikan dataset tersedia di folder `data/` dengan nama:

- `Fake.csv`
- `True.csv`

Jalankan training:

```bash
python train_model.py
```

Script ini akan membuat atau memperbarui file model di folder `models/` dan file evaluasi di folder `outputs/`.

## Menjalankan Aplikasi

Jalankan Streamlit:

```bash
streamlit run app.py
```

Setelah itu buka alamat lokal yang muncul di terminal, biasanya:

```text
http://localhost:8501
```

## Dataset

Dataset berisi dua kelas:

- `Fake.csv`: berita palsu
- `True.csv`: berita benar

Pada proses training, label yang digunakan adalah:

- Fake news = 1
- Real news = 0

## Catatan

Jika model belum tersedia atau file di folder `outputs/` belum ada, jalankan `train_model.py` terlebih dahulu sebelum membuka aplikasi Streamlit.
