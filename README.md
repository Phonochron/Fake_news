# Fake News Detection AI

Rencana kerja dan kriteria selesai tiap tahap ada di [docs/roadmap.md](docs/roadmap.md).

Aplikasi Streamlit untuk membandingkan prediksi Naive Bayes, Logistic Regression, Linear SVM, Voting Ensemble, dan LSTM pada teks berita berbahasa Inggris. Label `1` berarti fake/hoax dan `0` berarti real. Hasil akhir memakai model yang dipilih berdasarkan F1 validation dan dicatat dalam `models/selected_model.json`. Tabel hasil menampilkan `P(HOAX)` dari Naive Bayes, Logistic Regression, dan LSTM. Nilai tersebut belum dikalibrasi dan bukan tingkat kepastian bahwa suatu klaim salah; Linear SVM dan Voting Ensemble tidak menyediakan probabilitas. Artefak lama tanpa berkas pilihan model memakai mayoritas lima prediksi.

## Struktur

```text
fake_news/
  config.py           # Path dan konfigurasi bersama
  preprocessing.py    # Normalisasi teks yang dipakai training dan prediksi
  sequences.py        # Encoding dan padding bersama untuk LSTM
  data.py             # Pembacaan, audit, deduplikasi, dan pembagian dataset
  train.py            # Entrypoint training dengan urutan import yang aman
  classical_models.py # Training dan evaluasi model berbasis TF-IDF
  lstm_model.py       # Training dan evaluasi LSTM
  training.py         # Orkestrasi dan penyimpanan artefak
  evaluation.py       # Perhitungan metrik dan pemilihan model
  decision.py         # Perbandingan aturan keputusan pada validation
  reporting.py        # Tabel, grafik, dan laporan evaluasi
  provenance.py       # Hash dataset dan manifest artefak training
  explanations.py     # Kontribusi istilah untuk model linear
  prediction.py       # Pemuatan model dan prediksi
  dashboard.py        # Antarmuka Streamlit
  serve.py            # Launcher Streamlit untuk urutan import TensorFlow
  check_deployment.py # Pemeriksaan ukuran model dan waktu prediksi lokal
data/                # Fake.csv dan True.csv
models/              # Model dan tokenizer hasil training
outputs/             # Audit data, metrik, laporan, dan visualisasi
requirements.txt
```

`data/`, `models/`, dan `outputs/` tetap di lokasi semula agar artefak yang sudah ada dapat dipakai. Path artefak dihitung dari lokasi paket, sehingga pembacaan file tidak bergantung pada working directory.

## Menjalankan

Gunakan Python 3.11. Jalankan perintah ini dari direktori proyek:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m fake_news.serve
```

Jalankan pengujian dari direktori proyek dengan Python pada virtual environment yang sama:

```powershell
python -m pip check
python -m unittest discover -s tests -v
```

Pengujian mencakup preprocessing, deduplikasi dan pembagian dataset, cross validation tanpa kebocoran TF-IDF, seleksi berdasarkan validation, logika voting dan padding prediksi, serta tampilan Streamlit. Uji antarmuka memakai model tiruan sehingga tidak menimpa atau melatih ulang artefak.

Antarmuka menyediakan contoh artikel fiktif berbahasa Inggris tanpa label kebenaran. Pilih contoh itu pada menu **Input example**, lalu klik **Analyze News**; atau tempel artikel sendiri. Input dibatasi hingga 50.000 karakter. Teks kosong ditolak, sedangkan teks di bawah 30 kata tetap dapat dianalisis dengan peringatan bahwa hasilnya mungkin kurang andal. Teks panjang yang masih dalam batas diproses seperti input biasa; input melebihi batas ditolak oleh antarmuka dan fungsi prediksi. Hasil model adalah klasifikasi pola teks, bukan pemeriksaan fakta. Kinerja pada bahasa selain Inggris belum divalidasi.

Sebelum deployment, jalankan pemeriksaan artefak, ukuran model, dan waktu prediksi pada mesin target:

```powershell
python -m fake_news.check_deployment
```

Perintah ini memuat artefak sungguhan, menguji input pendek, biasa, dan hampir mencapai batas, lalu memastikan input kosong dan melebihi batas ditolak. Hasil waktu bergantung pada perangkat dan hanya menggambarkan pemeriksaan lokal. Catatan hasil saat pengembangan ada di [docs/deployment-readiness.md](docs/deployment-readiness.md).

Pada kombinasi dependensi Windows yang diuji, TensorFlow harus dimuat sebelum Streamlit agar DLL dapat diinisialisasi. Gunakan launcher `python -m fake_news.serve` untuk menjalankan aplikasi.

### Streamlit Community Cloud

Saat membuat aplikasi, gunakan repo `Phonochron/Fake_news`, branch `main`, main file path `fake_news/dashboard.py`, dan pilih **Python 3.11** melalui **Advanced settings**. Versi dependensi di `requirements.txt` diuji pada Python 3.11; memilih Python 3.14 dapat membuat instalasi gagal sebelum aplikasi berjalan. Untuk aplikasi yang sudah terbuat dengan Python 3.14, catat subdomain dan secrets yang dipakai, lalu hapus dan deploy ulang dengan Python 3.11. Streamlit Community Cloud tidak mengubah versi Python aplikasi yang sudah dibuat lewat perubahan `requirements.txt`. Korpus NLTK `stopwords` akan diunduh otomatis saat pertama kali diperlukan jika belum tersedia.

Untuk melatih ulang, pastikan `data/Fake.csv` dan `data/True.csv` tersedia, lalu jalankan:

```powershell
python -m fake_news.train
```

Training membaca kedua CSV, menggabungkan judul dan isi, lalu membuang artikel yang kosong, duplikat setelah normalisasi, atau memiliki label yang bertentangan sebelum pembagian data. Pembagian stratified menggunakan 70% train, 15% validation, dan 15% test dengan seed 42. TF-IDF dilatih di dalam setiap fold cross validation model klasik. LSTM menggunakan sebagian train untuk early stopping. Model dipilih sekali berdasarkan F1 validation; test dipakai setelah pemilihan untuk laporan akhir. Token eksplisit `Reuters` dihapus saat preprocessing karena sangat terkait dengan label pada dataset ini.

Training memeriksa kolom `title`, `text`, `subject`, dan `date` pada kedua CSV. Artikel yang kosong setelah preprocessing dibuang dan jumlah nilai kosong tiap kolom dicatat dalam audit. CSV yang tidak dapat dibaca, tidak memiliki kolom wajib, atau tidak menghasilkan dua kelas yang dapat dipakai akan gagal dengan pesan yang jelas.

Training menimpa berkas model di `models/` dan laporan di `outputs/`. Proses ini mahal karena mencakup cross validation dan LSTM. Jumlah fold dan epoch maksimum dapat diubah dengan `--cv-folds` dan `--lstm-epochs`. Laporan mencakup `evaluation_report.md`, `data_audit.json`, `subject_label_counts.csv`, `cv_metrics.csv`, `validation_metrics.csv`, dan `metrics.csv`. Skor CV adalah diagnostik training; pemilihan akhir tetap memakai validation terpisah.

Parameter lain tersedia lewat `--seed`, `--tfidf-max-features`, `--svm-c`, dan `--lstm-batch-size`; default dan parameter lain berada di `TrainingConfig` dalam `fake_news/config.py`. Setiap training menghasilkan `training_manifest.json` yang identik di `models/` dan `outputs/`, berisi hash kedua CSV, konfigurasi, seed, versi Python/library, model terpilih, metrik, dan hash artefak. Aplikasi memeriksa kecocokan kedua manifest saat memuat model. `decision_policy.json` menyimpan perbandingan F1 validation antara model terpilih dan voting lima model lama.

## Catatan hasil

Metrik di `outputs/metrics.csv` adalah hasil held-out test; kolom `Selected_On_Validation` menandai model yang dipilih sebelum test. Confusion matrix menampilkan model tersebut. Angka test tidak membuktikan kemampuan memeriksa kebenaran berita dari penerbit baru: data memiliki petunjuk kuat tentang sumber dan gaya penulisan, sedangkan identitas penerbit yang konsisten tidak tersedia untuk evaluasi terpisah berdasarkan sumber. Baca `outputs/evaluation_report.md` sebelum memakai skor sebagai klaim kinerja.

Voting akhir lima model tidak dipakai untuk artefak baru. Voting Ensemble sudah menggabungkan Naive Bayes, Logistic Regression, dan Linear SVM; menghitungnya lagi bersama ketiganya memberi suara berulang pada model yang berkorelasi. Aplikasi memakai model yang dipilih dari validation. Artefak lama tanpa `selected_model.json` tetap memakai voting lama demi kompatibilitas.

Pada training penuh terakhir, F1 validation Linear SVM adalah 0,9920, dibanding 0,9871 untuk voting lima model lama. Perbandingan ini memakai validation saja; test tidak dipakai untuk memilih aturan keputusan.

`outputs/top_hoax_words.csv` mengurutkan istilah berdasarkan selisih log likelihood Naive Bayes antara kelas HOAX dan REAL. Daftar ini menggambarkan hubungan pada dataset, bukan alasan untuk prediksi artikel tertentu atau penjelasan model Linear SVM yang dipilih.

Untuk model linear terpilih (Linear SVM atau Logistic Regression), aplikasi menampilkan istilah dari artikel yang kontribusinya paling besar pada skor model. Tanda positif mengarah ke kelas HOAX dan tanda negatif ke REAL. Istilah telah melalui preprocessing dan kontribusi ini bukan bukti faktual tentang artikel. Model lain tidak memiliki penjelasan istilah dengan metode ini.

Model `*.pkl` yang disertakan dibuat dengan scikit-learn 1.9.0. `requirements.txt` mengunci versi dependensi utama yang dipakai saat uji prediksi pada Python 3.11. Jika lingkungan lama memakai versi berbeda, buat virtual environment baru dan pasang dependensi kembali, atau latih ulang agar seluruh artefak konsisten. Aplikasi memberi pesan jika versi model tidak cocok.
