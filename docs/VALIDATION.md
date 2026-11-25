# Status verifikasi — pembaruan 29 Oktober 2026

Perbaikan training diuji menggunakan Python 3.14.6 dan torch 2.14.1+cpu yang terpasang di komputer pengguna.

- 10 tes pytest lulus: preprocessing audio/resampling/stereo/keheningan, pembaruan bobot kedua varian CNN, agregasi loss per lagu, penolakan kebocoran group dan duplikat byte, serta aturan checkpoint/early stopping.
- Demo baseline dengan patience 0 dan monitor loss menyelesaikan 30 epoch. Loss validasi terakhir sekitar 0.6816 dan hasil test sintetis 3/6 benar. Menambah epoch saja belum memberikan pemisahan stabil.
- Demo frequency dengan patience 0 dan monitor loss menyelesaikan 30 epoch. Checkpoint terpilih epoch 30, loss validasi sekitar 0.0090, hasil test sintetis 6/6 benar.
- Artefak run yang diuji disalin ke runs/demo-fixed pada proyek pengguna. report.json mempertahankan args eksekusi asli dan mencatat pemindahan artefak.
- GPU belum diuji: paket yang tersedia adalah PyTorch CPU. Saat diperiksa, .venv proyek hanya memiliki pip; gunakan interpreter yang dijelaskan dalam TRAINING_FIX.md.
- Belum dilakukan: training/evaluasi musik asli dan fusion hybrid. Hasil nada sintetis bukan klaim kualitas deteksi musik AI.

Lihat TRAINING_FIX.md untuk perintah yang dapat diulang dengan folder output baru.
