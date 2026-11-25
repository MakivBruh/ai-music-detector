# Hybrid Music Forensics — baseline tugas akhir

Proyek Python/PyTorch lokal untuk **melatih CNN dari bobot acak**. Tidak menggunakan API inferensi, embedding eksternal, atau bobot pralatih. Asumsi lingkup awal: deteksi musik manusia (0) versus musik yang dihasilkan AI (1). Audio campuran manusia/AI belum menjadi kelas tersendiri. Nama “hybrid” mengacu pada rencana menggabungkan CNN dan fitur akustik; fusion belum diimplementasikan pada tahap baseline ini.

## Windows: mulai di sini

Python 3.11 bukan syarat khusus kode ini. Gunakan Python 64-bit dengan versi paket yang kompatibel; perbaikan demo diuji dengan Python 3.14 dan PyTorch CPU yang sudah terpasang. Jalankan PowerShell dari folder proyek ini. Contoh pembuatan environment baru di bawah memakai Python 3.14; sesuaikan dengan interpreter yang terpasang.

```powershell
py -3.14 -m venv .venv
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

Demo membuat 18 rekaman nada sintetis. Label hanya membedakan frekuensi nada. Akurasi demo **bukan bukti deteksi musik AI**. Output folder harus baru; gunakan nama lain untuk mengulang. Tes mencakup preprocessing, update bobot nyata, agregasi per lagu, penolakan kebocoran group, augmentasi codec (deterministik per seed, kedua kelas train terkena, val/test tidak), balanced sampler, dan pembuat manifest MUSEBENCH (pasangan Jamendo–generasi dalam satu group, LOGO). `pytest.ini` membatasi pencarian tes ke `tests/` agar salinan di `backups/` tidak ikut terkumpul.

Untuk memeriksa apakah model benar-benar dapat mempelajari demo, lihat `docs/TRAINING_FIX.md`. Gunakan `--model frequency --monitor loss --patience 0 --epochs 30`: posisi frekuensi dipertahankan, checkpoint dipilih dari loss validasi terendah, dan early stopping dimatikan. Cache demo lama dapat digunakan kembali. Pengaturan ini merupakan eksperimen diagnostik dan harus ditetapkan terpisah sebelum penelitian memakai data asli.

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

Default di atas tetap berlaku untuk `--model baseline`. Pilihan `--model frequency` mempertahankan 8 posisi frekuensi dan merata-ratakan waktu. `--monitor loss` menggunakan binary cross entropy per lagu dari rata-rata probabilitas segmennya; loss ini berbeda dari loss training per segmen dengan dropout aktif. `--patience N` mengatur toleransi epoch tanpa perbaikan (`0` berarti tidak berhenti lebih awal). `--min-delta` menentukan besar peningkatan minimum. Jangan memilih pengaturan dari hasil test musik asli.

## Mencegah shortcut codec

Run `hmf-28000` (SONICS vs FMA-small) mencapai akurasi test 99,7%, tetapi skornya mengikuti **kualitas file**, bukan asal-usul musik: semua lagu AI berupa MP3 berkualitas rendah dengan potongan frekuensi ~7,3 kHz, sedangkan semua lagu manusia berupa MP3 ~300 kbps. Lagu FMA yang dikompres ulang ke MP3 mono 16 kHz 32 kbps naik dari skor 0,00 ke 0,86. Perbaikan di bawah membuat codec tidak lagi berkorelasi dengan label.

**1. Augmentasi codec untuk kedua kelas** (`hmf/audio.py`, `hmf/prepare.py`). `--augment-variants K` membuat K versi fitur per lagu train. Setiap versi melewati rantai ffmpeg acak (ffmpeg dari `imageio_ffmpeg`): gain −12…+4 dB, lowpass 4–7,9 kHz (peluang 40%), resample 16–48 kHz, mono/stereo, lalu MP3/AAC/Opus 32–320 kbps atau PCM 16-bit, kemudian didekode kembali. Rantai ditentukan oleh `seed`, sha256 file dan nomor varian, sehingga hasilnya dapat diulang dan tidak bergantung pada urutan atau jumlah worker. Val/test tetap satu versi bersih. `--loudness-norm` menormalkan RMS setiap jendela 5 detik ke −20 dBFS; pengaturan ini disimpan di `audio_config.json`, sehingga `predict`, studio web dan evaluasi memakai normalisasi yang sama. `--fmax 7000` membatasi log-mel hingga 7 kHz. Pada pilot, bin mel teratas (~7,6 kHz) bernilai sekitar −46 dB untuk lagu FMA dan −76 dB untuk lagu AI SONICS. Setelah lagu FMA dikompres ke MP3 32 kbps, nilainya turun menjadi −78 dB. Augmentasi dapat mengurangi bandwidth lagu manusia, tetapi tidak dapat menambahkan bandwidth ke lagu AI. Karena itu, pita tersebut tetap menjadi petunjuk label hingga disembunyikan dari model. Opsi lain: `--workers N` (paralel), `--float16` (separuh ruang disk), `--path-map LAMA=BARU` (memetakan path Windows di manifest ke lokasi lain, misalnya Colab), `--skip-missing`.

**2. Sampling seimbang** (`hmf/train.py`). Default `--balance sampler` memakai `WeightedRandomSampler` sehingga setiap kelas mendapat setengah dari sampel per epoch (train lama 17.478 AI vs 5.600 human). `--balance-by generator` juga memberi porsi sama bagi setiap generator/sumber dalam satu kelas. `--balance weight` memakai class weight di loss; `--balance none` mengikuti perilaku lama. `report.json` kini mencatat akurasi dan rata-rata skor test per generator dan per sumber (`per_generator`, `per_source`).

**3. Manifest multi-sumber MUSEBENCH** (`hmf/manifest.py`). Unduh subset dengan `scripts/musebench_index.py fetch`, lalu:

```powershell
.\.venv\Scripts\python.exe -m hmf.manifest musebench --root D:\musebench --reference-manifest data/manifest-28000.csv
```

Label diberikan berdasarkan folder: `Jamendo/` dan `OOD FMA/` = manusia; `<nama>_generated_music/` = AI dengan `generator` musicgen, audioldm, riffusion, magnet, jasco, magenta atau stable_audio. `group_id` diturunkan dari nama lagu dasar (`Lagu - Artis.mp3.wav` → `lagu - artis`). Dengan cara ini lagu Jamendo dan semua generasi dari folder `Jamedo/` yang memakai nama sama berada dalam satu group dan satu split. Generasi `Genius-Lyrics/` dari lirik yang sama juga dikelompokkan. `OOD FMA/<id>_...` mendapat group `fma_<id>` seperti FMA di manifest SONICS. `--reference-manifest` membuat group yang sudah ada di manifest lain mempertahankan split-nya. Folder perturbasi (aac/eq/noise/loudness/pitch) hanya berisi lagu AI yang diubah, sehingga perubahan itu sendiri bisa menjadi petunjuk label baru. Karena itu, folder tersebut dilewati secara default, begitu juga `ai_human_co-created_music` (label ambigu) dan suno/udio (sudah ada di SONICS). Generasi MUSEBENCH berformat WAV 32 kHz, sedangkan lagu manusia berformat MP3 44,1 kHz; augmentasi pada poin 1 diterapkan ke keduanya. Manifest dapat digabungkan: `hmf.prepare --manifest data/manifest-28000.csv D:\musebench\manifest-musebench.csv`.

**4. Leave-one-generator-out.** Generator yang ditahan hanya muncul di test, bersama lagu Jamendo pasangannya. Lagu AI dari generator lain dibuang dari test, sehingga skor test hanya mengukur generator yang tidak pernah dilihat. Pemeriksaan menolak group lintas split dan split tanpa salah satu label.

```powershell
.\.venv\Scripts\python.exe -m hmf.manifest logo --manifest data/manifest-28000.csv D:\musebench\manifest-musebench.csv --holdout musicgen --output data/manifest-logo-musicgen.csv
```

**5. Evaluasi OOD dan robustness** (`hmf/evaluate_folder.py`). Siapkan `data/ood/ai/` (lagu AI baru, misalnya Suno/Udio versi terbaru yang tidak ada di SONICS) dan `data/ood/human/` (lagu manusia di luar FMA). Setiap lagu diskor dalam kondisi bersih dan setelah dikompres ulang: `mp3_32k_mono16k`, `mp3_128k`, `aac_128k`, `opus_64k`. Hasilnya `per_track.csv` (skor dan selisih per lagu) dan `summary.json` (akurasi, balanced accuracy, AUC, rata-rata skor per kelas, |Δ| rata-rata/maksimum, dan *flip rate*). Beberapa run dapat dibandingkan sekaligus:

```powershell
.\.venv\Scripts\python.exe -m hmf.evaluate_folder --run runs/hmf-28000 runs/hmf-aug-v1 --ai data/ood/ai --human data/ood/human --max-segments 12 --output runs/ood-compare
```

Model yang tidak memakai shortcut codec seharusnya memiliki |Δ| kecil dan flip rate mendekati 0. Pada tiga lagu FMA dari pilot, `hmf-28000` mendapat skor `mp3_32k_mono16k` 0,50–1,00 (clean 0,00). Uji pilot-1000 (2 varian, 2 epoch CPU, 20+20 lagu test): tanpa `--fmax`, skor human di `mp3_32k_mono16k` naik dari 0,04 ke 0,91 (flip 50%). Dengan `--fmax 7000`, |Δ| ≤ 0,01 dan flip 0% untuk semua kondisi. Akurasi bersih turun ke ~0,78, yang lebih jujur karena model tidak lagi memakai petunjuk codec. Angka ini berasal dari pilot kecil, bukan hasil akhir.

**6. Training di Google Colab** (`colab/train.ipynb`). Training penuh tidak dijalankan di laptop. Unggah `dist/ai-music-detector-colab.zip` (kode + `manifest-28000.csv`), `fake_songs.zip` dan `fma_small.zip` ke `MyDrive/HMF/`, lalu sesuaikan sel konfigurasi pertama bila nama berbeda. Pekerjaan dibagi dua sesi dan bisa dicicil. **Sesi CPU** (tidak butuh kuota GPU): notebook menyalin audio ke `/content`, membuat indeks MUSEBENCH sekali (disimpan di Drive), memilih subset yang muat di disk beserta lagu Jamendo pasangannya, membuat manifest (opsional LOGO), lalu menjalankan `prepare --resume --max-new-tracks N` berulang. Setiap folder 100 lagu yang selesai disimpan ke `Drive/HMF/cache/<RUN_NAME>/parts/` sebagai tar, bersama `plan.json` (daftar lagu tetap) dan sampel lagu test untuk tes robustness. Bila sesi putus, jalankan notebook lagi dari atas: cache dipulihkan dari Drive dan audio hanya disiapkan untuk lagu yang belum selesai. **Sesi GPU**: jalankan notebook dari atas; cache dipulihkan, unduhan dilewati, lalu training (output ke `Drive/HMF/runs/<RUN_NAME>`), tes robustness dan evaluasi OOD dibandingkan model lama. Cache di Drive butuh ±20–25 GB. Di CLI, `--resume` melanjutkan folder yang punya `plan.json`, `--max-new-tracks N` berhenti setelah N lagu baru (exit code 3 = sebagian selesai), dan `--drop-duplicates` menyimpan salinan pertama file byte-identik (mis. OOD FMA MUSEBENCH = FMA-small). Hasil dicicil identik dengan sekali jalan karena rantai augmentasi ditentukan oleh seed dan sha256 file. Training juga bisa dilanjutkan: `hmf.train` menyimpan `last.pt` (bobot, optimizer, scaler) di setiap epoch, dan `--resume` melanjutkan dari epoch terakhir yang tersimpan; untuk run lama tanpa `last.pt`, lanjutan dimulai dari `best.pt` dengan optimizer baru. Early stopping dipulihkan dari `history.json`. `--skip-train-eval` melewati penilaian ulang data train tiap epoch (epoch ±2x lebih cepat). Run yang terhenti dapat diberi `report.json` sementara dengan `python -m hmf.finalize` (stop_reason `interrupted`) agar bisa dibuka di studio web; report itu otomatis diganti saat training lanjutan selesai.

Uji cepat lokal tanpa training penuh (pilot 1000 lagu, 2 epoch CPU):

```powershell
.\.venv\Scripts\python.exe -m hmf.prepare --manifest data/pilot-1000/manifest.csv --output data/cache-pilot-aug --augment-variants 2 --max-segments 2 --loudness-norm --fmax 7000 --workers 6 --float16
.\.venv\Scripts\python.exe -m hmf.train --cache data/cache-pilot-aug --output runs/pilot-aug --epochs 2 --cpu --model frequency --monitor loss
.\.venv\Scripts\python.exe -m hmf.evaluate_folder --run runs/pilot-aug --ai data/ood/ai --human data/ood/human --max-segments 6
```

Lisensi MUSEBENCH: secara umum CC BY-SA 3.0, Jamendo mengikuti lisensi per lagu, dan output generator sebagian besar non-komersial. Simpan atribusi dan jangan mendistribusikan ulang audio. Aturan tugas akhir tetap berlaku: tidak ada bobot pralatih.

## RTX 3050 4 GB / RAM 16 GB

Default batch 8, pekerja pembaca 0 agar sederhana di Windows, mixed precision saat CUDA tersedia. Jika memori tidak cukup, turunkan batch ke 4 lalu 2 dan tutup aplikasi yang memakai GPU. Snapshot lama menunjukkan sekitar 2.7 GB VRAM sudah terpakai; itu bukan pengukuran kondisi sekarang. Jangan menjanjikan batch tertentu akan muat. Preprocessing tidak memerlukan GPU; cache memakai disk sekitar 80 KB per segmen (sekitar 58 MB per jam audio, di luar overhead file).

Training menyimpan `best.pt` dan `report.json` berisi argumen, konfigurasi audio, versi torch, loss, metrik validasi dan prediksi test per track. `best.pt` hanya state_dict, gunakan arsitektur CNN yang sama. Simpan manifest, source audio dengan hak akses sesuai lisensi, commit kode dan environment-lock bersama hasil penelitian. Seed mengurangi variasi, tidak menjamin determinisme lintas perangkat.

Perbaikan training menambahkan `history.json` setiap epoch, metrik `train_eval` tanpa dropout, `best_epoch`, `epochs_completed`, dan `stop_reason`. Perintah predict membaca pilihan model dari report; report lama memakai baseline secara otomatis. Jangan memisahkan best.pt dari report.json miliknya.

## Batas interpretasi

Model mengestimasi pola pada dataset; skor bukan pembuktian asal-usul sebuah karya. Label harus berasal dari provenance yang terdokumentasi, bukan tebakan pendengaran. Data berhak cipta tidak disertakan. Lihat `docs/EVALUATION.md` untuk rancangan penelitian dan tahap hybrid.

Rujukan implementasi: [instalasi resmi PyTorch](https://pytorch.org/get-started/locally/) dan [automatic mixed precision](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

## Studio web lokal

Dari folder proyek, jalankan `.\.venv\Scripts\python.exe -m hmf.web`, lalu buka http://127.0.0.1:8765 di browser. Server memilih `best.pt` terbaru dari run yang memiliki `report.json` dan menampilkan nama run aktif di pojok kanan atas halaman; pilih model lain dengan `--run`. Jika port sudah dipakai, server menolak berjalan (pakai `--port` lain). Tekan Ctrl+C untuk berhenti.

Pilih file audio (WAV, MP3, M4A, OGG, FLAC, WMA, dan lainnya) hingga 32 MB, lalu klik **Analisis rekaman**. Hasilnya meliputi:
- vonis, skor AI, nilai `ai_score` persis, dan meter terhadap ambang;
- spektrogram log-mel yang dilihat model, dengan skor tiap segmen 5 detik di bawahnya pada sumbu waktu yang sama (hover untuk detail, klik untuk memilih segmen dan memindahkan pemutar);
- JSON yang bisa dilihat atau diunduh (tanpa data gambar).

**Umpan balik dan uji lapangan.** Setelah vonis muncul, tandai **Benar**, **Salah**, atau **Tidak tahu**, lalu pilih dari mana kamu tahu label lagu itu (wajib untuk Benar/Salah). Umpan balik disimpan lokal di `data/feedback/feedback.jsonl` (hash, skor, vonis, label, run; jawaban terakhir untuk lagu dan model yang sama yang dipakai). Audio hanya disalin ke `data/feedback/audio/<label>/` bila kotak izinnya dicentang. Panel **Uji lapangan** menampilkan akurasi dengan rentang 95% (Wilson), salah tuduh, lagu AI yang lolos, dan daftar kasus yang salah. Ringkasan dan ekspor lewat CLI: `python -m hmf.feedback report [--csv out.csv]`, `python -m hmf.feedback export --run <run> --only-wrong --ood data/ood` (untuk `hmf.evaluate_folder`) atau `--manifest data/feedback/manifest.csv --split train` (untuk melatih ulang). Ini uji lapangan, bukan test resmi: lagu dipilih pengguna dan labelnya berasal dari pengakuan; lagu yang dipakai melatih ulang tidak boleh dipakai menguji model yang sama. Folder lain bisa dipilih dengan `--feedback-dir`.

**Melatih ulang dengan lagu tambahan tanpa prepare ulang.** `hmf.train --cache <utama> <tambahan> ...` melatih dari beberapa cache sekaligus (cache pertama = utama). Cache tambahan dibuat dari manifest kecil (mis. ekspor umpan balik) dengan pengaturan audio yang sama dan `--partial-splits`, yang mengizinkan cache hanya berisi split train atau satu kelas. Pengaturan audio semua cache harus identik; lagu (sha256) yang sudah ada di cache sebelumnya dilewati agar lagu test tidak bocor ke training. Dengan `--balance-by generator`, sumber `feedback` mendapat porsi sampel sendiri. Di notebook, isi `EXTRA_CACHES` dengan folder atau zip cache tambahan di Drive. `hmf.finalize` menerima daftar cache yang sama.

Bagian **Tentang model** (endpoint `/model`) menampilkan metrik test, kurva loss train/validasi, akurasi per generator, dan konfigurasi audio run aktif. Halaman mendukung mode gelap dan layar ponsel. Buku manual lengkap ada di `docs/manual/` (PDF dan Word; screenshot di `docs/manual/img/`).
