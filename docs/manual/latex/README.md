# Panduan Template LaTeX Laprak dan Makalah

## Peta berkas dan alur kerja

```text
latex-academic-template/
├── main.tex                 # Identitas dan urutan dokumen; kompilasi berkas ini
├── preamble.tex             # Font, margin, penomoran, paket, dan format
├── chapters/
│   ├── 00-cover.tex
│   ├── 00-kata-pengantar.tex
│   ├── 01-pendahuluan.tex
│   ├── 02-dasar-teori.tex
│   ├── 03-metode.tex
│   ├── 04-panduan-penggunaan.tex
│   ├── 05-hasil-pembahasan.tex
│   ├── 06-penutup.tex
│   └── 07-lampiran.tex
├── figures/                 # Logo, gambar, grafik
├── references.bib           # Data sumber referensi
└── README.md                # Panduan ini
```

**Yang sering diedit:** identitas di main.tex, isi di chapters/, sumber di references.bib. Aturan tampilan cukup diubah di preamble.tex. Berkas bab tidak perlu documentclass, begin{document}, atau end{document} sendiri.

Perintah \input{chapters/01-pendahuluan} memasukkan isi berkas bab ke dokumen utama. Untuk menambah bab, buat berkas .tex di chapters/ dan tambahkan satu baris \input{chapters/nama-berkas} pada posisi yang tepat di main.tex. Untuk menghapus bab dari dokumen, komentari baris input-nya dengan %. Penomoran otomatis mengikuti urutannya; nomor pada nama berkas hanya untuk merapikan folder.

Semua jalur gambar dan input dihitung dari folder main.tex, bukan dari folder chapters/. Selalu kompilasi main.tex.

## Mulai dalam 5 langkah

1. **Salin seluruh folder latex-academic-template** untuk setiap tugas baru. Jangan hanya menyalin main.tex jika sudah menambahkan gambar.
2. Buka main.tex. Ganti teks bertanda [kurung siku] pada bagian **IDENTITAS**. Nama anggota cukup diubah pada AnggotaKelompok.
3. Masukkan logo ke folder figures dengan nama logo.png. Ukuran logo diatur melalui LebarLogo (bawaan 8 cm).
4. Buka berkas yang sesuai di chapters/, cari komentar **ISI:** dan tulis isi laporan tepat di bawahnya. Komentar diawali tanda persen dan tidak muncul di PDF.
5. Kompilasi memakai **LuaLaTeX** atau **XeLaTeX**. Untuk sitasi .bib, ikuti urutan kompilasi pada bagian 8.

Di editor LaTeX Codex: buka folder proyek atau main.tex, edit, lalu periksa pratinjau PDF.
Di Overleaf: unggah seluruh folder sebagai ZIP, jadikan main.tex dokumen utama, lalu pilih LuaLaTeX atau XeLaTeX pada pengaturan compiler.
Di komputer dengan instalasi LaTeX: jalankan LuaLaTeX pada main.tex; ikuti urutan kompilasi pada bagian 8 untuk memperbarui daftar isi dan sitasi.

Font memakai Times New Roman bila tersedia. Jika tidak tersedia, template memakai TeX Gyre Termes yang bentuknya serupa. Untuk syarat **Times New Roman persis**, gunakan komputer/compiler yang memiliki font tersebut.

## Yang sudah diatur

- A4, margin kiri 4 cm; kanan, atas, bawah 3 cm.
- Isi 12 pt, judul bab 14 pt tebal; tulis judul bab dengan huruf kapital.
- Isi 1,5 spasi, indentasi paragraf 1 cm, rata kiri-kanan.
- Caption 10 pt dan spasi tunggal; daftar pustaka spasi tunggal.
- Nomor halaman awal Romawi kecil dan isi utama angka Arab.
- Nomor bab Romawi, subbab/tabel/gambar mengikuti nomor bab.
- Cover, kata pengantar, daftar isi, daftar gambar, daftar tabel, lima bab, daftar pustaka, lampiran.

Isi laporan sengaja kosong. Petunjuk berada di komentar sumber agar tidak ikut tercetak. Hapus bagian yang tidak diperlukan sesuai instruksi dosen; struktur ini bukan ketentuan wajib semua institusi.

## 1. Identitas dan anggota

Ganti bagian di dalam kurung kurawal kedua, bukan nama perintahnya:

~~~latex
\newcommand{\JudulLaporan}{PENGUJIAN GERBANG LOGIKA}
\newcommand{\DosenPengampu}{Nama Dosen, S.T., M.T.}
~~~

Setiap baris anggota memiliki tiga kolom. Tanda & memisahkan kolom dan \\ mengakhiri baris:

~~~latex
1 & Nama Mahasiswa & 1234567890\\
2 & Nama Mahasiswa Lain & 1234567891\\
~~~

Tambahkan atau hapus baris sesuai jumlah anggota. Bila nama atau judul panjang, periksa cover agar tidak melewati satu halaman.

## 2. Bab dan subbab

~~~latex
\chapter{PENDAHULUAN}
\section{Latar Belakang}
\subsection{Permasalahan yang Diamati}
~~~

Nomor dibuat otomatis. Jangan mengetik “BAB I” atau “1.1” sendiri.
Judul tanpa nomor memakai tanda bintang:

~~~latex
\chapter*{KATA PENGANTAR}
\addcontentsline{toc}{chapter}{KATA PENGANTAR}
~~~

Baris kedua memasukkan judul tersebut ke daftar isi. Untuk bab biasa, jangan tambahkan baris itu karena daftar isi sudah otomatis.

## 3. Paragraf, penekanan, dan daftar

Ketik teks biasa. Sisakan **satu baris kosong** untuk membuat paragraf baru.
Jangan memakai \\ untuk memisahkan paragraf isi; perintah itu untuk pindah baris tertentu, misalnya di cover atau tabel.

~~~latex
Ini paragraf pertama.

Ini paragraf kedua dengan \textbf{teks tebal} dan \textit{teks miring}.
~~~

Daftar bernomor untuk langkah kerja:

~~~latex
\begin{enumerate}
\item Menyiapkan alat dan bahan.
\item Melakukan pengamatan.
\item Mencatat hasil.
\end{enumerate}
~~~

Daftar tanpa nomor:

~~~latex
\begin{itemize}
\item Breadboard.
\item Kabel penghubung.
\end{itemize}
~~~

## 4. Tabel pengamatan

Letakkan judul tabel **di atas** tabel. Contoh berikut hanya contoh bentuk, bukan data hasil praktikum:

~~~latex
\begin{table}[htbp]
\centering
\singlespacing
\caption{Hasil pengamatan.}
\label{tab:pengamatan}
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{cccc}
\toprule
No. & Masukan & Teori & Pengamatan\\
\midrule
1 & -- & -- & --\\
2 & -- & -- & --\\
\bottomrule
\end{tabular}
\end{table}
~~~

- c = kolom rata tengah; l = kiri; r = kanan.
- Empat huruf c berarti empat kolom dan tiap baris membutuhkan tiga tanda &.
- Untuk kolom keterangan yang panjang, gunakan p{5cm} atau tabularx.
- Jangan memasukkan hasil perkiraan sebagai hasil pengamatan.

Tabel dengan kolom keterangan yang dapat membungkus teks:

~~~latex
\begin{table}[htbp]
\centering
\singlespacing
\caption{Alat dan bahan.}
\label{tab:alat}
\begin{tabularx}{\textwidth}{c X c}
\toprule
No. & Alat atau bahan & Jumlah\\
\midrule
1 & Nama alat dan spesifikasi & --\\
\bottomrule
\end{tabularx}
\end{table}
~~~

Untuk tabel panjang yang melewati halaman, gunakan longtable **tanpa** lingkungan table:

~~~latex
{\singlespacing
\begin{longtable}{c p{8cm} c}
\caption{Data lengkap.}\label{tab:lengkap}\\
\toprule No. & Keterangan & Nilai\\ \midrule
\endfirsthead
\toprule No. & Keterangan & Nilai\\ \midrule
\endhead
\bottomrule
\endfoot
1 & Keterangan pengamatan & --\\
% Tambahkan baris data di sini.
\end{longtable}
}
~~~

## 5. Gambar dan dokumentasi

Simpan gambar, misalnya figures/rangkaian.png, kemudian gunakan:

~~~latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{rangkaian.png}
\caption{Susunan rangkaian praktikum.}
\label{fig:rangkaian}
\end{figure}
~~~

Folder figures sudah ditentukan di template. Format PNG, JPG, dan PDF dapat digunakan.
Nama file sebaiknya sederhana: huruf, angka, tanda hubung; hindari spasi.
Judul gambar berada **di bawah** gambar. Ganti 0.85 menjadi 0.7 atau 1 untuk mengatur lebarnya.

Dua gambar berdampingan:

~~~latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.47\textwidth]{kondisi-1.png}\hfill
\includegraphics[width=0.47\textwidth]{kondisi-2.png}
\caption{Kondisi pertama (kiri) dan kondisi kedua (kanan).}
\label{fig:dua-kondisi}
\end{figure}
~~~

[htbp] membolehkan LaTeX mencari posisi yang rapi. Jika harus tepat di lokasi kode, gunakan [H], tetapi gambar besar dapat menyisakan ruang kosong.
Gunakan \FloatBarrier setelah satu kelompok pembahasan agar gambar/tabel tidak melewati batas itu. Jangan memaksa semua gambar dengan [H] bila hasilnya terlalu renggang.

## 6. Persamaan dan simbol

Persamaan di dalam kalimat menggunakan $...$:

~~~latex
Tegangan dinyatakan sebagai $V=I R$.
~~~

Persamaan terpisah tanpa nomor:

~~~latex
\[
Y=\overline{A\cdot B}
\]
~~~

Persamaan bernomor:

~~~latex
\begin{equation}
R=\frac{V}{I}
\label{eq:resistansi}
\end{equation}
~~~

Beberapa baris yang sejajar:

~~~latex
\begin{align}
Y &= \overline{A\cdot A}\\
  &= \overline{A}
\end{align}
~~~

Perintah yang sering dipakai:

| Keperluan | Kode dalam mode matematika |
|---|---|
| Pecahan | \frac{a}{b} |
| Pangkat | x^{2} |
| Indeks bawah | V_{\mathrm{out}} |
| Akar | \sqrt{x} |
| Garis atas / negasi | \overline{A\cdot B} |
| Perkalian | \times atau \cdot |
| Huruf Yunani | \alpha, \beta, \Delta |
| Ohm | \Omega |
| Derajat | 30^\circ |
| Satuan tegangan | 5\,\mathrm{V} |
| Persen dalam rumus | 10\% |

Jelaskan arti simbol dan satuan setelah persamaan. Untuk NAND, garis di atas seluruh hasil perkalian ditulis \overline{A\cdot B}.

## 7. Rujukan silang otomatis

Letakkan \label tepat setelah \caption, atau di dalam persamaan bernomor.

~~~latex
Hasil dicantumkan pada Tabel~\ref{tab:pengamatan}.
Susunan alat ditunjukkan pada Gambar~\ref{fig:rangkaian}.
Perhitungan menggunakan Persamaan~\eqref{eq:resistansi}.
~~~

Setiap label harus unik. Gunakan awalan tab:, fig:, eq:, atau sec: agar mudah dicari.
Jika keluar tanda ??, periksa nama label lalu kompilasi ulang dua kali.

## 8. Daftar pustaka otomatis dari references.bib

Isi sumber di references.bib, bukan di dalam bab. Contoh entri ada di file tersebut sebagai komentar. Hapus tanda % pada satu blok contoh dan ganti seluruh data dengan sumber yang benar-benar dipakai. Tahun dan nama dalam contoh bukan sumber nyata.

~~~bibtex
@book{sumber1,
  author = {NamaBelakang, NamaDepan},
  title = {Judul Buku},
  year = {2026},
  publisher = {Nama Penerbit}
}
~~~

Kutip di bab menggunakan perintah berikut:

~~~latex
Pembahasan mengacu pada sumber tersebut \cite{sumber1}.
Dua sumber dapat dikutip bersama \cite{sumber1,modul}.
~~~

Kunci sumber1 harus sama dengan kunci setelah @book{ di references.bib dan harus unik. Hanya sumber yang dikutip yang otomatis muncul. Untuk menyertakan satu sumber tanpa kutipan di kalimat, gunakan \nocite{modul} sebelum daftar pustaka jika sesuai pedoman dosen.

Buku manual ini memakai biblatex dengan backend Biber dan gaya numerik (backend BibTeX bawaan template memunculkan peringatan "rerun BibTeX" yang tidak hilang). Jangan mencampurnya dengan \bibitem, thebibliography, atau \bibliography. Judul daftar pustaka sudah tersedia di main.tex; jangan ditambah lagi.

**Urutan kompilasi lokal setelah mengisi sumber:**

~~~text
lualatex main.tex
biber main
lualatex main.tex
lualatex main.tex
~~~

Jalankan dari folder proyek. Jika memakai XeLaTeX, ganti lualatex dengan xelatex. Overleaf biasanya mengurus tahap Biber secara otomatis. Jika editor lain tidak menjalankan BibTeX, jalankan urutan di atas dengan instalasi LaTeX lokal atau gunakan fitur build bibliography editor tersebut. Paket biblatex dan program BibTeX harus tersedia.

Selama template belum memiliki sumber atau kutipan, daftar pustaka kosong dan peringatan empty bibliography dapat muncul; ini wajar. Setelah berisi, tanda [?] atau kunci sumber berarti kunci kutipan salah atau tahap BibTeX belum selesai.

Untuk modul Classroom, masukkan identitas bibliografisnya tanpa URL restricted. Jangan mengarang data yang tidak tersedia.

## 9. Lampiran, catatan kaki, dan halaman baru

Isi lampiran pada bagian yang telah disediakan. Tambahkan judul dengan pola berikut:

~~~latex
\section*{Lampiran C. Kode Program}
~~~

Catatan kaki:

~~~latex
Kalimat yang memerlukan keterangan.\footnote{Isi keterangan tambahan.}
~~~

\newpage memulai halaman baru; \clearpage juga mengeluarkan gambar/tabel yang masih tertunda.
Bab baru otomatis dimulai di halaman baru, sehingga umumnya tidak perlu menambah \newpage sebelum setiap bab.

## 10. Karakter khusus dan kesalahan umum

| Ingin menulis | Ketik |
|---|---|
| Persen | \% |
| Ampersand | \& |
| Garis bawah dalam teks | \_ |
| Pagar | \# |
| Dolar | \$ |
| Kurung kurawal | \{ dan \} |

Tanda % biasa memulai komentar sehingga sisa baris tidak dicetak.
Karakter _ biasa dipakai untuk indeks di dalam rumus, bukan teks biasa.
Tanda & biasa dipakai untuk pemisah kolom tabel atau titik sejajar dalam align.

Jika kompilasi gagal:
- **Fontspec requires XeTeX or LuaTeX:** ganti compiler ke LuaLaTeX/XeLaTeX.
- **File not found:** cek nama file, ekstensi, dan folder gambar.
- **Missing $ inserted:** cek tanda _ atau rumus yang belum diapit $.
- **Extra alignment tab:** jumlah tanda & tidak sesuai jumlah kolom tabel.
- **Undefined control sequence:** cek ejaan perintah atau paket yang dibutuhkan.
- **Missing } / runaway argument:** cari kurung kurawal yang belum ditutup.
- **Teks keluar margin:** pendekkan judul, bungkus kolom dengan p{...}/X, atau kecilkan lebar gambar.

## 11. Mengubah menjadi makalah

Salin folder terlebih dahulu, lalu:
- Ganti JenisDokumen menjadi MAKALAH dan kosongkan NomorPercobaan bila tidak dipakai.
- Pertahankan pendahuluan, rumusan masalah, dan tujuan; ubah Tujuan Praktikum menjadi Tujuan Penulisan.
- Susun bab isi sesuai materi, misalnya BAB II TINJAUAN PUSTAKA dan BAB III PEMBAHASAN.
- Hapus metode, hasil pengamatan, pengolahan data, serta lampiran yang tidak relevan.
- Akhiri dengan PENUTUP dan DAFTAR PUSTAKA.
- Hapus daftar gambar/tabel jika tidak ada gambar/tabel.

Penomoran akan mengikuti perubahan secara otomatis setelah dikompilasi ulang.

## Sebelum dikumpulkan

Cari semua tanda [ dan komentar ISI untuk memastikan tidak ada isian tertinggal. Periksa judul, nama, NIM, dosen, tahun, isi daftar pustaka, rujukan, ukuran gambar, dan halaman cover. Gunakan hasil praktikum yang benar-benar diperoleh. Simpan PDF hasil akhir bersama sumber .tex dan folder gambar.

