# Pemeriksaan kesiapan lokal

Tahap produk menyediakan contoh input fiktif berbahasa Inggris, menjelaskan bahwa label adalah prediksi pola teks, memperingatkan input di bawah 30 kata, dan membatasi input hingga 50.000 karakter. Aplikasi belum divalidasi untuk bahasa lain dan tidak memverifikasi kebenaran suatu klaim.

Jalankan pemeriksaan pada mesin yang akan dipakai untuk deployment:

```powershell
python -m fake_news.check_deployment
```

Perintah tersebut memverifikasi kecocokan manifest dan hash artefak, memuat model sebenarnya, mengukur waktu impor TensorFlow dan pemuatan model, lalu memprediksi input pendek, biasa, dan panjang. Input kosong dan input melewati batas harus ditolak. Ukuran tiap berkas model dan totalnya dicetak sebagai JSON. Semua waktu adalah pengukuran lokal, bukan jaminan latensi di hosting.

Pada pemeriksaan lokal 29 September 2026, artefak dengan `run_id` `75e2ef71-b66a-4910-ba8a-8b6a742514f3` berukuran total 24.708.292 byte. Impor TensorFlow memerlukan 20,322 detik, pemuatan model 2,537 detik, dan prediksi pertama untuk teks pendek 1,262 detik. Prediksi teks biasa memerlukan 0,213 detik dan teks panjang 0,319 detik. Teks kosong serta teks melebihi batas ditolak. Pengukuran ini perlu diulang pada perangkat target karena proses awal, memori, CPU, dan lalu lintas pengguna akan berbeda.

Pengujian antarmuka dengan model tiruan mencakup input kosong, contoh input, hasil prediksi, serta pesan kegagalan manifest. Pemeriksaan server lokal juga mencapai `script_finished` melalui koneksi Streamlit tanpa exception. Verifikasi visual di browser dan pengujian beban serentak masih perlu dilakukan pada lingkungan deployment yang dipilih.
