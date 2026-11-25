# Memperbaiki demo yang berhenti di epoch 6

Pada laporan sebelumnya, validation macro-F1 sama selama enam epoch. Epoch 1 dipilih dan training berhenti setelah lima epoch tanpa peningkatan. Menambah batas epoch saja tidak menonaktifkan kondisi ini.

Perbaikan menyediakan opsi patience, monitor dan min-delta; loss validasi per lagu; ringkasan epoch; histori tersimpan setiap epoch; dan metadata pemilihan checkpoint. Default macro-F1/patience 5 dipertahankan untuk kompatibilitas. Gunakan pengaturan diagnostik di bawah untuk demo.

Uji baseline 30 epoch dengan early stopping dimatikan menunjukkan loss hanya turun sedikit dan belum memisahkan kedua kelas secara stabil. Demo membedakan rentang frekuensi nada; pooling global awal mengurangi informasi posisi frekuensi. Varian frequency mempertahankan delapan posisi frekuensi setelah konvolusi dan merata-ratakan waktu. Arsitektur baseline masih tersedia untuk perbandingan. Ini bukan bukti bahwa varian baru lebih baik pada musik asli.

## Jalankan dari proyek yang benar

```powershell
cd "C:\Users\makiv\OneDrive\Documents\Coding\projek\ai-music-detector"
& "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\python.exe" -m hmf.train --cache data/cache-demo --output runs/demo-fixed-2 --epochs 30 --batch-size 8 --model frequency --monitor loss --patience 0 --cpu
```

Interpreter di atas sudah memiliki torch 2.14.1+cpu di komputer ini. Saat diperiksa, .venv proyek hanya memiliki pip. Karena itu perintah tersebut memakai Python utama yang paketnya tersedia. Jika nanti memakai environment lain, ganti interpreter dengan path environment yang benar dan cek impor torch terlebih dahulu. Tidak perlu mengubah dependency untuk mencoba perbaikan ini.

Folder output harus baru; ganti angka akhiran jika sudah ada. Cache demo lama dapat dipakai. Model dilatih ulang dari bobot acak. Pada report.json periksa:

- `epochs_completed`: 30 jika proses selesai penuh.
- `stop_reason`: max_epochs karena patience 0.
- `best_epoch`: epoch dengan loss validasi terbaik, tidak harus epoch terakhir.
- `history[*].train_eval` dan `history[*].val`: loss/akurasi per lagu, tanpa dropout.
- `history[*].train_loss`: loss optimasi per segmen saat dropout aktif, sehingga tidak persis sebanding dengan loss evaluasi.

`hmf.predict` memilih arsitektur sesuai report.json. Checkpoint baseline lama masih dapat dibaca. Model baru harus digunakan bersama report dari run yang sama.

## Pengujian

Tes regresi mencakup plateau epoch 6, patience 0 sampai 30 epoch, pemilihan loss saat F1 sama, validasi parameter, loss berbobot sama per track, preprocessing, gradient update untuk kedua varian CNN, kebocoran group dan duplikat byte. Hasil demo sintetis digunakan untuk memeriksa pipeline, bukan untuk klaim akurasi deteksi musik AI.

## Hasil verifikasi aktual

10 tes lulus dengan Python 3.14.6 / torch 2.14.1+cpu. Demo frequency selesai 30 epoch; checkpoint epoch 30; validation loss 0.0090; test sintetis 6/6 benar. Hasil uji dipindahkan dari folder validasi ke runs/demo-fixed; args asli tetap tersimpan pada report.json. Cache yang dipakai berasal dari proyek ini. Cadangan kode sebelum perubahan tersedia di backups/.
