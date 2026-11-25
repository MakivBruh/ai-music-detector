# Status verifikasi — 29 Oktober 2026

- Lulus: parsing sintaks seluruh 8 file Python menggunakan Python 3.12.14.
- Lulus: struktur contoh manifest, kedua kelas tersedia pada masing-masing split.
- Perangkat terdeteksi: NVIDIA GeForce RTX 3050 Laptop GPU, VRAM total 4096 MiB; snapshot pemakaian 2440 MiB. Ini bukan pengukuran kebutuhan training.
- Belum terverifikasi: empat tes pytest, preprocessing audio aktual, pembaruan bobot CNN, training end-to-end, prediksi file, dan CUDA. Dependensi PyTorch/librosa belum tersedia pada awal sesi; unduhan paket uji berjalan terlalu lambat untuk menuntaskan uji runtime. Tidak ada klaim tes runtime lulus.
- Belum dilakukan: training dengan musik asli, evaluasi kualitas deteksi dan implementasi fusion hybrid. Dataset asli tidak disertakan.

## Menuntaskan verifikasi

Ikuti instalasi pada README, kemudian jalankan `python -m pytest -q` dan rangkaian demo menggunakan interpreter environment tersebut. Tes memeriksa bentuk log-mel, resampling stereo/keheningan, update parameter, agregasi per track, kebocoran group, dan duplikat byte. Setelah demo training, uji prediksi:

```powershell
.\.venv\Scripts\python.exe -m hmf.predict --run runs/demo --audio data/demo/test_0_0.wav
```

Keberhasilan demo hanya memverifikasi alur perangkat lunak. Simpan output aktual dan versi dependensi sebelum mengklaim hasil penelitian.
