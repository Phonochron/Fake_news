# Roadmap proyek

Ruang lingkup lima tahap ini mengikuti rencana yang disepakati dengan pemilik proyek.

| Tahap | Fokus | Kriteria selesai | Status |
| --- | --- | --- | --- |
| 1. Lingkungan dan pengujian | Lingkungan bersih, model lama termuat, uji prediksi dan Streamlit | Prediksi nyata dan semua uji lulus | Selesai |
| 2. Evaluasi model | TF-IDF di dalam CV, pemilihan via validation, test akhir, audit duplikat dan sumber | Laporan split, pilihan model, dan keterbatasan | Selesai |
| 3. Logika prediksi | Bandingkan voting lima model dengan model pilihan validation; jelaskan probabilitas model | Aturan keputusan terdokumentasi dan diuji; UI tidak mengklaim verifikasi fakta | Selesai |
| 4. Data dan training | Validasi CSV, konfigurasi, seed, versi library dan dataset, kata pembeda kelas | Satu perintah menghasilkan model, metrik, dan metadata yang cocok | Selesai |
| 5. Produk dan deployment | Contoh input, batas bahasa Inggris, input kosong/pendek/panjang, performa dan ukuran model | Pengguna memahami hasil dan aplikasi stabil | Pemeriksaan lokal selesai; verifikasi pada target deployment menunggu lingkungan target |

Perbaikan struktur kode pada tahap 3 dan beberapa elemen antarmuka pada tahap 4 dikerjakan lebih awal. Status di atas mengikuti kriteria rencana ini, bukan urutan tanggal perubahan file.

Pada training penuh terakhir, Linear SVM mendapat F1 validation 0,9920 dan voting lima model lama 0,9871. Artefak yang terpasang memiliki `run_id` yang sama pada `models/training_manifest.json` dan `outputs/training_manifest.json`; hash seluruh berkas model dan laporan sudah diverifikasi.
