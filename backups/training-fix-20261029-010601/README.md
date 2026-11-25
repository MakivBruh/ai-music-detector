# Hybrid Music Forensics — baseline tugas akhir

Proyek Python/PyTorch lokal untuk **melatih CNN dari bobot acak**. Tidak menggunakan API inferensi, embedding eksternal, atau bobot pralatih. Asumsi lingkup awal: deteksi musik manusia (0) versus musik yang dihasilkan AI (1). Audio campuran manusia/AI belum menjadi kelas tersendiri. Nama “hybrid” mengacu pada rencana menggabungkan CNN dan fitur akustik; fusion belum diimplementasikan pada tahap baseline ini.

## Windows: mulai di sini

Gunakan Python 3.11 64-bit. Jalankan PowerShell dari folder proyek ini. `python` yang membuka Microsoft Store berarti Python belum tersedia pada PATH; gunakan launcher `py -3.11` setelah memasang Python.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Pasang wheel PyTorch GPU lebih dahulu: buka https://pytorch.org/get-started/locally/ dan pilih Windows, Pip, Python, CUDA yang didukung. Jalankan perintah yang diberikan, mengganti `pip` dengan `.\.venv\Scripts\python.exe -m pip`. Kemudian:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.\.venv\Scripts\python.exe -m pip freeze > environment-lock.txt
```

Angka CUDA di `nvidia-smi` bukan perintah memilih versi toolkit/wheel. Verifikasi dengan PyTorch di environment tersebut. Jika CUDA False, perbaiki pemasangan wheel GPU; training tetap dapat berjalan pada CPU. Dependency diberi rentang kompatibilitas, bukan klaim bahwa semua kombinasi telah diuji; simpan hasil freeze untuk setiap eksperimen.

## Uji alur tanpa dataset asli

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m hmf.demo --output data/demo
.\.venv\Scripts\python.exe -m hmf.prepare --manifest data/demo/manifest.csv --output data/cache-demo
.\.venv\Scripts\python.exe -m hmf.train --cache data/cache-demo --output runs/demo --epochs 2 --cpu
```

Demo membuat 18 rekaman nada sintetis. Label hanya membedakan frekuensi nada. Akurasi demo **bukan bukti deteksi musik AI**. Output folder harus baru; gunakan nama lain untuk mengulang. Tes mencakup preprocessing, update bobot nyata, agregasi per lagu, dan penolakan kebocoran group.

## Dataset asli

Salin `data/manifest.example.csv` menjadi `data/manifest.csv`, ganti semua baris dengan rekaman sebenarnya. Path audio relatif terhadap folder manifest. Utamakan WAV/FLAC; konversi format lain secara konsisten sebelum membuat manifest. Kolom wajib: `path,label,group_id,split`; simpan pula source, generator, versi, lisensi, prompt, tanggal, genre dan durasi jika tersedia. Contoh manifest hanya struktur, bukan dataset yang disertakan.

Pilih split pada tingkat karya/sumber sebelum pemotongan, misalnya 70/15/15 berdasarkan kelompok sambil menjaga kedua kelas pada setiap split. Semua versi lagu, remix, excerpt dan turunan karya yang sama harus mempunyai group_id sama. Untuk hasil generator dengan keluarga prompt/reference sama, kelompokkan juga agar tidak bocor. Pemeriksaan otomatis menolak group lintas split dan file byte-identik; duplikat transkode membutuhkan audit tambahan. Jangan menganggap hash file cukup untuk mendeteksi kemiripan audio.

```powershell
.\.venv\Scripts\python.exe -m hmf.prepare --manifest data/manifest.csv --output data/cache-v1
.\.venv\Scripts\python.exe -m hmf.train --cache data/cache-v1 --output runs/cnn-seed42 --epochs 30 --batch-size 8 --seed 42
.\.venv\Scripts\python.exe -m hmf.predict --run runs/cnn-seed42 --audio data/raw/contoh.wav
```

Preprocessing di CPU, dibaca per jendela: stereo menjadi mono, resample 16 kHz, potong nonoverlap 5 detik, pad potongan terakhir, log-mel 64 bin, FFT 1024/hop 256, skala dB tetap. Tanpa normalisasi menggunakan statistik test. Cache float32 dimuat per sampel, bukan seluruh dataset ke RAM. Potongan terakhir yang pendek tetap dipakai; analisis sensitivitas harus membandingkan kebijakan ini dengan membuang ekor pendek. 16 kHz menghilangkan komponen di atas 8 kHz, sehingga artefak frekuensi tinggi tidak dapat dinilai oleh baseline ini.

CNN 3 blok konvolusi (16/32/64), global pooling dan dropout. Cross entropy, AdamW, learning rate 0.001; checkpoint terbaik berdasarkan macro-F1 **per lagu pada validation**, early stop setelah 5 epoch tanpa perbaikan. Probabilitas segmen dirata-rata per lagu; threshold 0.5 ditetapkan di awal. Test hanya dievaluasi setelah pemilihan checkpoint. Lagu panjang menghasilkan lebih banyak sampel training; laporkan distribusi durasi dan gunakan ablation sampling seimbang bila perlu.

## RTX 3050 4 GB / RAM 16 GB

Default batch 8, pekerja pembaca 0 agar sederhana di Windows, mixed precision saat CUDA tersedia. Jika memori tidak cukup, turunkan batch ke 4 lalu 2 dan tutup aplikasi yang memakai GPU. Snapshot lama menunjukkan sekitar 2.7 GB VRAM sudah terpakai; itu bukan pengukuran kondisi sekarang. Jangan menjanjikan batch tertentu akan muat. Preprocessing tidak memerlukan GPU; cache memakai disk sekitar 80 KB per segmen (sekitar 58 MB per jam audio, di luar overhead file).

Training menyimpan `best.pt` dan `report.json` berisi argumen, konfigurasi audio, versi torch, loss, metrik validasi dan prediksi test per track. `best.pt` hanya state_dict, gunakan arsitektur CNN yang sama. Simpan manifest, source audio dengan hak akses sesuai lisensi, commit kode dan environment-lock bersama hasil penelitian. Seed mengurangi variasi, tidak menjamin determinisme lintas perangkat.

## Batas interpretasi

Model mengestimasi pola pada dataset; skor bukan pembuktian asal-usul sebuah karya. Label harus berasal dari provenance yang terdokumentasi, bukan tebakan pendengaran. Data berhak cipta tidak disertakan. Lihat `docs/EVALUATION.md` untuk rancangan penelitian dan tahap hybrid.

Rujukan implementasi: [instalasi resmi PyTorch](https://pytorch.org/get-started/locally/) dan [automatic mixed precision](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
