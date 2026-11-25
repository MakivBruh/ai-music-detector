# Rencana evaluasi

## Pertanyaan penelitian

Apakah gabungan representasi log-mel yang dipelajari CNN dan fitur akustik eksplisit meningkatkan kemampuan deteksi musik AI pada generator yang tidak terlihat saat training? Baseline yang disertakan baru menjawab bagian CNN; klaim hybrid menunggu implementasi dan eksperimen fusion.

## Dataset dan kontrol

Dokumentasikan jumlah track, jam, genre, bahasa, instrumental/vokal, generator/versi dan provenance setiap kelas. Sediakan kelas manusia dan AI dari beberapa sumber, dengan codec, sample rate dan distribusi durasi yang sebanding. Hindari label sepenuhnya identik dengan platform/codec: classifier bisa belajar watermark, encoder, loudness atau keheningan. Gunakan hanya data yang izin penelitiannya jelas.

Split group berdasarkan karya/keluarga prompt; jangan split segmen secara acak. Audit fingerprint audio untuk duplikat transkode. Manifest harus dibekukan sebelum eksperimen. Kedua kelas harus ada di train/val/test. Jumlah data ditentukan berdasarkan ketersediaan dan analisis learning curve; jangan memakai enam contoh manifest sebagai ukuran dataset penelitian.

## Urutan eksperimen

1. Jalankan demo sebagai pemeriksaan teknis, lalu periksa manual beberapa fitur dari data asli dan labelnya.
2. Jalankan CNN pada split tetap dengan seed 42, 43, 44. Pilih hiperparameter hanya dari validation; tetapkan anggaran pencarian sebelum membuka hasil test.
3. Tambahkan baseline fitur MFCC mean/std, spectral centroid/rolloff/flatness, zero-crossing rate dan RMS dengan logistic regression atau SVM. Fit scaler hanya pada train.
4. Implementasikan hybrid: gabungkan embedding CNN dengan vektor fitur terstandarisasi, latih head fusion memakai train. Bandingkan CNN saja, fitur saja dan fusion pada split dan anggaran sama. Tetapkan apakah encoder dibekukan atau dilatih bersama sebelum eksperimen.
5. Uji generalisasi dengan menahan satu generator sepenuhnya dari train/val, lalu ulangi untuk generator lain. Gunakan manifest terpisah dan audit group lagi.
6. Uji robustness pada test yang ditransformasi: codec, resampling, perubahan gain dan noise. Semua turunan tetap milik group test asal. Jangan memilih checkpoint dari robustness test.

## Pelaporan

Unit utama adalah lagu/karya, bukan segmen. Laporkan macro-F1 sebagai metrik utama, balanced accuracy, ROC-AUC, confusion matrix (baris label asli, kolom prediksi; urutan manusia=0, AI=1), serta precision/recall tiap kelas. Semua metrik ini disimpan dalam report.json. Threshold default 0.5; bila dituning, gunakan validation saja dan simpan nilainya. Skor softmax belum terkalibrasi sebagai probabilitas forensik.

Laporkan rerata dan simpangan baku antar seed. Tambahkan interval kepercayaan bootstrap pada tingkat group/karya (bukan segmen) dan paired bootstrap untuk membandingkan model pada karya yang sama. Laporkan jumlah group pada tiap split, termasuk saat sampel terlalu sedikit untuk interval stabil. Sertakan hasil per genre, durasi, sumber dan generator; gunakan metadata manifest untuk menghubungkan track_id (nomor baris mulai 0) pada report.

Catat runtime, penggunaan VRAM puncak, ukuran parameter, ruang disk dan kegagalan OOM pada laptop sebenarnya. Ukuran batch dan setting mixed precision harus dilaporkan. Jangan memberi klaim keberhasilan penelitian berdasarkan demo sintetis. Audio campuran/editing parsial, atribusi generator dan lokalisasi segmen manipulasi adalah pekerjaan lanjutan dengan label dan evaluasi tersendiri.
