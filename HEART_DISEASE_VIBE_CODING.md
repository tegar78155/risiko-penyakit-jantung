# Vibe Coding Brief — Heart Disease Classification dengan Logistic Regression

Dokumen ini adalah spesifikasi proyek dan kumpulan prompt untuk membangun TA-3 menggunakan GitHub Copilot dan Kilo Code. Gunakan dokumen ini sebagai *single source of truth* selama pengembangan. Jangan mengubah tujuan proyek atau mengganti dataset tanpa persetujuan.

---

## 1. Ringkasan proyek

**Judul:** Implementasi Algoritma Logistic Regression untuk Klasifikasi Indikasi Penyakit Jantung Berdasarkan Dataset Cleveland Berbasis Web.

**Tujuan:** Membuat aplikasi web akademik yang menerima masukan atribut pasien, menjalankan model Logistic Regression yang telah dilatih, lalu menampilkan prediksi dua kelas beserta probabilitas model.

**Kelas target:**
- `0` — Tidak ada indikasi penyakit jantung dalam definisi label dataset.
- `1` — Ada indikasi penyakit jantung dalam definisi label dataset.

**Catatan penting:** Aplikasi ini hanya prototipe pembelajaran mesin untuk kepentingan akademik. Aplikasi bukan alat diagnosis, skrining klinis, atau pengganti pemeriksaan dokter. Gunakan frasa **“indikasi menurut model”**, bukan “pasien pasti sakit/sehat”. Jangan memberikan saran pengobatan.

**Metode utama:** Logistic Regression dari Scikit-learn.

**Teknologi yang disarankan:**
- Python 3.10 atau lebih baru
- Pandas dan NumPy
- Scikit-learn
- Matplotlib dan Seaborn
- Flask
- Joblib
- HTML, CSS, dan Bootstrap (opsional)
- Pytest (opsional, untuk pengujian)

---

## 2. Dataset yang benar-benar dikirim

File yang diunggah bernama `heart+disease.zip`. ZIP tersebut berisi beberapa dataset, antara lain Cleveland, Hungarian, Switzerland, dan Long Beach VA, serta file dokumentasi. **Untuk proyek ini gunakan hanya `processed.cleveland.data` sebagai dataset utama. Jangan menggabungkan seluruh file `.data` dan jangan menggunakan `cleveland.data` mentah sebagai pengganti tanpa alasan dan persetujuan.**

### File utama

`processed.cleveland.data`

Hasil pemeriksaan file:
- 303 baris data.
- 14 kolom tanpa header.
- 13 kolom fitur dan 1 kolom target.
- Ada 6 nilai hilang yang ditulis sebagai tanda `?`.
- Nilai hilang tersebut berada pada kolom `ca` (4 nilai) dan `thal` (2 nilai).

Ekstrak ZIP, lalu buat folder `data/` di root proyek. Salin file `processed.cleveland.data` ke:

```text
data/processed.cleveland.data
```

Jangan mengedit file sumber secara manual. Buat salinan hasil preprocessing hanya bila memang diperlukan.

### Urutan kolom dataset

File tidak memiliki header. Berikan nama kolom berdasarkan urutan berikut:

| Urutan | Nama kolom | Arti | Perlakuan awal |
|---:|---|---|---|
| 1 | `age` | Usia dalam tahun | Numerik kontinu |
| 2 | `sex` | Jenis kelamin; 1 = laki-laki, 0 = perempuan | Kategorikal |
| 3 | `cp` | Jenis nyeri dada; kode 1–4 | Kategorikal |
| 4 | `trestbps` | Tekanan darah istirahat (mm Hg) | Numerik kontinu |
| 5 | `chol` | Kolesterol serum (mg/dl) | Numerik kontinu |
| 6 | `fbs` | Gula darah puasa > 120 mg/dl; 1 = ya, 0 = tidak | Kategorikal/biner |
| 7 | `restecg` | Hasil ECG saat istirahat; kode 0–2 | Kategorikal |
| 8 | `thalach` | Detak jantung maksimum yang dicapai | Numerik kontinu |
| 9 | `exang` | Angina akibat olahraga; 1 = ya, 0 = tidak | Kategorikal/biner |
| 10 | `oldpeak` | Depresi ST akibat olahraga dibandingkan kondisi istirahat | Numerik kontinu |
| 11 | `slope` | Kemiringan segmen ST puncak olahraga; kode 1–3 | Kategorikal |
| 12 | `ca` | Jumlah pembuluh darah utama yang terlihat melalui fluoroskopi; kode 0–3 | Kategorikal; memiliki nilai hilang |
| 13 | `thal` | Kategori thal; kode 3, 6, atau 7 pada dokumentasi dataset | Kategorikal; memiliki nilai hilang |
| 14 | `num` | Target asli: tingkat/keberadaan penyakit jantung, kode 0–4 | Target |

Keterangan kode `cp` menurut dokumentasi dataset:
- 1 = typical angina
- 2 = atypical angina
- 3 = non-anginal pain
- 4 = asymptomatic

Keterangan `restecg`:
- 0 = normal
- 1 = kelainan gelombang ST-T
- 2 = kemungkinan/definite hipertrofi ventrikel kiri menurut kriteria Estes

Keterangan `slope`:
- 1 = upsloping
- 2 = flat
- 3 = downsloping

Keterangan `thal`:
- 3 = normal
- 6 = fixed defect
- 7 = reversible defect

### Transformasi target

Dokumentasi dataset menjelaskan bahwa target `num` bernilai 0 sampai 4. Untuk klasifikasi biner, definisikan:

```python
 df["target"] = (df["num"] > 0).astype(int)
```

Artinya:
- `num == 0` menjadi `target == 0`.
- `num` bernilai 1, 2, 3, atau 4 menjadi `target == 1`.

Setelah membuat `target`, keluarkan `num` dari fitur agar tidak terjadi **target leakage**. Jangan memakai `num` maupun `target` sebagai input fitur model.

Distribusi target asli dalam dokumentasi file:
- `num = 0`: 164 baris.
- `num = 1`: 55 baris.
- `num = 2`: 36 baris.
- `num = 3`: 35 baris.
- `num = 4`: 13 baris.

Dengan pengelompokan biner di atas, distribusi menjadi:
- Tidak ada indikasi (`0`): 164 baris.
- Ada indikasi (`1`): 139 baris.

### Catatan kualitas data

- Saat memuat file, gunakan `na_values="?"` agar tanda `?` dibaca sebagai missing value.
- Pastikan seluruh 14 kolom terbaca dan kolom numerik berhasil dikonversi ke numerik.
- Validasi jumlah baris, jumlah kolom, missing value, duplikasi, rentang nilai, dan distribusi target.
- Jangan mengisi missing value sebelum pembagian train/test. Imputasi harus menjadi bagian dari pipeline yang hanya di-*fit* pada data training.
- Jangan menghapus data hanya untuk mempercantik hasil evaluasi. Laporkan keputusan preprocessing secara transparan.

---

## 3. Ruang lingkup aplikasi

Bangun aplikasi Flask sederhana, responsif, dan mudah dipahami. Antarmuka sebaiknya berbahasa Indonesia.

### Halaman yang dibutuhkan

1. **Beranda**
   - Nama proyek dan penjelasan singkat.
   - Tujuan aplikasi.
   - Peringatan bahwa hasil bukan diagnosis medis.
   - Tombol menuju halaman prediksi dan informasi model.

2. **Halaman Prediksi**
   - Form input untuk 13 fitur: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.
   - Gunakan pilihan dropdown/radio untuk fitur kategorikal agar pengguna tidak memasukkan kode kategori sembarangan.
   - Gunakan input angka dengan validasi batas wajar untuk fitur numerik. Batas validasi harus didokumentasikan dan tidak boleh diklaim sebagai batas diagnosis medis.
   - Validasi semua kolom wajib di server, bukan hanya HTML.
   - Setelah submit, tampilkan kelas prediksi dan probabilitas untuk kedua kelas.

3. **Halaman Tentang Model**
   - Penjelasan Logistic Regression secara ringkas.
   - Dataset dan jumlah sampel.
   - Daftar fitur dan target.
   - Cara mengubah `num` menjadi target biner.
   - Ringkasan metrik evaluasi dari hasil pelatihan aktual.
   - Tampilkan tanggal/waktu pelatihan bila tersedia.

4. **Halaman Hasil Prediksi** (boleh berupa bagian pada halaman prediksi)
   - Label hasil: “Ada indikasi menurut model” atau “Tidak ada indikasi menurut model”.
   - Probabilitas kelas 0 dan kelas 1 dalam persen.
   - Jelaskan bahwa probabilitas merupakan keluaran model, bukan probabilitas klinis yang tervalidasi.
   - Tampilkan disclaimer medis.

### Fitur tambahan (opsional)

- Grafik distribusi target.
- Grafik Confusion Matrix.
- Grafik ROC Curve.
- Ringkasan metrik Accuracy, Precision, Recall, F1-Score, dan ROC-AUC.
- Tombol untuk mengosongkan form.

Jangan menambahkan fitur yang tidak diperlukan sebelum alur utama berhasil.

---

## 4. Ketentuan pemodelan dan evaluasi

### Pemisahan fitur dan target

- Fitur (`X`) adalah 13 kolom selain `num` dan `target`.
- Target (`y`) adalah kolom `target` biner.
- Lakukan `train_test_split` dengan `test_size=0.2` atau `0.25`, `random_state=42`, dan `stratify=y`.
- Jangan melakukan preprocessing dengan mempelajari seluruh data sebelum split.

### Preprocessing yang disarankan

Gunakan `Pipeline` dan `ColumnTransformer` Scikit-learn agar preprocessing yang sama dipakai saat training dan prediksi web.

Kelompok fitur yang disarankan:

**Numerik kontinu:**
- `age`
- `trestbps`
- `chol`
- `thalach`
- `oldpeak`

Preprocessing:
- `SimpleImputer(strategy="median")`
- `StandardScaler()`

**Kategorikal:**
- `sex`
- `cp`
- `fbs`
- `restecg`
- `exang`
- `slope`
- `ca`
- `thal`

Preprocessing:
- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

Model:
- `LogisticRegression(max_iter=2000, class_weight=None, random_state=42)` sebagai baseline awal.
- Jika ada alasan untuk mengubah parameter atau memakai `class_weight="balanced"`, bandingkan hasilnya secara terukur dan jelaskan alasannya.
- Jangan langsung memakai SMOTE. Dataset biner memiliki distribusi kelas yang relatif tidak timpang (164 vs 139); mulai dari baseline tanpa SMOTE.
- Jika ingin mencoba SMOTE sebagai eksperimen tambahan, hanya lakukan di dalam pipeline/tiap fold data training menggunakan `imblearn`, bukan sebelum train/test split.

### Evaluasi wajib

Tampilkan dan jelaskan:
- Confusion Matrix.
- Accuracy.
- Precision.
- Recall.
- F1-Score.
- ROC-AUC.
- Classification Report.
- ROC Curve.

Gunakan `predict_proba` untuk menghitung ROC-AUC dan menampilkan probabilitas. Pastikan kelas positif didefinisikan sebagai `target = 1`.

Karena data hanya 303 baris, jangan mengklaim model sangat akurat atau siap dipakai secara klinis. Sebaiknya tambahkan `StratifiedKFold` cross-validation (misalnya 5 fold) pada data training sebagai analisis tambahan. Jelaskan bahwa skor dapat berubah karena ukuran dataset kecil.

Jangan menjanjikan skor tertentu. Jangan memanipulasi test set, threshold, atau pembagian data hanya untuk mendapatkan akurasi tinggi. Laporkan hasil aktual, termasuk bila performanya rendah.

### Penyimpanan model

- Simpan keseluruhan pipeline (imputer + scaler/encoder + Logistic Regression) sebagai satu file Joblib, misalnya `artifacts/heart_disease_pipeline.joblib`.
- Jangan menyimpan model terpisah dari preprocessing jika hal itu membuat prediksi web memakai transformasi yang berbeda.
- Simpan metadata fitur, urutan fitur, kelas target, dan metrik evaluasi dalam file JSON terpisah, misalnya `artifacts/model_metadata.json`.
- Aplikasi harus memuat model hasil training; jangan melatih ulang model setiap kali halaman web dibuka.
- Jika file model belum tersedia, tampilkan pesan yang jelas dan arahkan pengembang menjalankan skrip training.

---

## 5. Struktur folder yang disarankan

```text
heart-disease-logreg/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .python-version
├── .gitignore
├── data/
│   └── processed.cleveland.data
├── artifacts/
│   ├── heart_disease_pipeline.joblib
│   └── model_metadata.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── evaluation.py
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   └── about.html
├── public/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   └── images/
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── target_distribution.png
└── tests/
    ├── test_data.py
    ├── test_model.py
    └── test_app.py
```

Struktur ini boleh disederhanakan jika memang lebih cocok untuk proyek kecil, tetapi pisahkan minimal kode training, aplikasi web, data, dan artifacts model.

---

## 6. Persyaratan teknis dan kualitas kode

- Kode harus dapat dijalankan di Windows melalui VS Code.
- Gunakan `pathlib` untuk mengelola path file agar tidak bergantung pada current working directory.
- Buat `requirements.txt` dengan library yang benar-benar digunakan.
- Gunakan fungsi terpisah untuk load data, preprocessing, training, evaluasi, dan prediksi.
- Tambahkan error handling untuk file dataset/model yang tidak ditemukan, nilai input tidak valid, dan kegagalan prediksi.
- Jangan menaruh data pribadi atau kredensial di repository.
- Tambahkan `data/`, file cache, virtual environment, dan file sementara ke `.gitignore` bila diperlukan. Pastikan kebijakan lisensi/distribusi dataset diperiksa sebelum mengunggah dataset ke repository publik.
- Jangan membuat angka metrik contoh seolah-olah hasil aktual. Semua angka pada halaman web harus dibaca dari hasil training/metadata.
- Pastikan urutan nama kolom input ke pipeline selalu konsisten.
- Gunakan HTML escaping bawaan Jinja; jangan menampilkan input mentah sebagai HTML.
- Berikan komentar seperlunya, bukan komentar untuk setiap baris.
- Jangan menambah database, autentikasi, API eksternal, atau framework frontend besar kecuali diminta.

---

## 7. Acceptance criteria — kapan proyek dianggap selesai

Proyek dinyatakan selesai hanya bila seluruh poin berikut terpenuhi:

- [ ] Aplikasi membaca `data/processed.cleveland.data` dengan benar.
- [ ] Dataset terbaca menjadi 303 baris dan 14 kolom sebelum transformasi target.
- [ ] Missing value `?` dikenali sebagai missing value.
- [ ] Kolom diberi nama sesuai dokumentasi dan urutannya benar.
- [ ] Target biner dibuat dari `num > 0`.
- [ ] Kolom `num` dan `target` tidak masuk ke fitur model.
- [ ] Split training/testing dilakukan dengan stratifikasi.
- [ ] Preprocessing di-fit hanya pada training data melalui pipeline.
- [ ] Logistic Regression berhasil dilatih.
- [ ] Metrik evaluasi dan grafik dibuat dari data uji yang tidak digunakan untuk fitting.
- [ ] Pipeline tersimpan dengan Joblib dan dapat dimuat kembali.
- [ ] Form Flask menerima seluruh 13 fitur dan memvalidasi input di server.
- [ ] Prediksi web konsisten dengan prediksi pipeline yang sama.
- [ ] Probabilitas dua kelas ditampilkan dengan benar.
- [ ] Disclaimer akademik/non-diagnostik terlihat jelas.
- [ ] README menjelaskan instalasi, training, menjalankan aplikasi, struktur folder, dataset, serta keterbatasan.
- [ ] Tidak ada error saat menjalankan alur lengkap dari training sampai prediksi web.

---

# 8. Prompt utama untuk Copilot atau Kilo Code

Salin prompt berikut ke Copilot Chat atau Kilo Code setelah membuka folder proyek di VS Code. Sebaiknya mulai dari mode Agent, tetapi minta AI membuat rencana dahulu sebelum mengubah banyak file.

```text
Bertindaklah sebagai senior Python machine learning engineer dan Flask developer yang membantu saya mengerjakan proyek akademik TA-3.

Baca seluruh file HEART_DISEASE_VIBE_CODING.md ini terlebih dahulu dan jadikan isinya sebagai spesifikasi utama proyek. Jangan mengabaikan aturan dataset, target, preprocessing, evaluasi, dan batasan aplikasi yang tertulis di dalamnya.

TUJUAN PROYEK:
Membangun aplikasi web berbahasa Indonesia untuk klasifikasi indikasi penyakit jantung menggunakan Logistic Regression dan dataset Cleveland yang saya unggah.

DATASET:
Gunakan hanya data/processed.cleveland.data. File tidak memiliki header, memiliki 303 baris dan 14 kolom. Berikan nama kolom sesuai urutan di dokumen ini. Baca tanda '?' sebagai missing value. Target asli adalah num (0–4); buat target biner dengan aturan num == 0 menjadi 0 dan num > 0 menjadi 1. Jangan memasukkan num atau target ke dalam fitur model.

ATURAN PENTING:
1. Jangan membuat dataset sintetis atau mengganti dataset.
2. Jangan mengarang hasil metrik atau mengklaim skor tertentu.
3. Jangan melakukan preprocessing sebelum train/test split. Gunakan Pipeline dan ColumnTransformer agar imputasi, scaling, dan encoding hanya di-fit pada data training.
4. Gunakan stratified train/test split dan Logistic Regression sebagai model utama.
5. Mulai tanpa SMOTE; dataset biner relatif seimbang.
6. Evaluasi dengan Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC-AUC, Classification Report, serta ROC Curve.
7. Simpan satu pipeline lengkap memakai Joblib agar preprocessing saat inferensi sama dengan preprocessing saat training.
8. Buat aplikasi Flask dengan form 13 fitur, validasi server-side, prediksi kelas, dan probabilitas kedua kelas.
9. Tampilkan disclaimer bahwa hasil hanya untuk pembelajaran dan bukan diagnosis medis.
10. Gunakan path yang aman dan portabel dengan pathlib. Pastikan kode berjalan pada Windows.
11. Jangan menambahkan database, login, API eksternal, atau teknologi yang tidak diperlukan.

CARA KERJA:
Tahap 1 — INSPEKSI DAN RENCANA
- Periksa struktur folder dan file dataset yang tersedia.
- Tampilkan rencana implementasi, daftar file yang akan dibuat/diubah, dan risiko teknis yang ditemukan.
- Jangan langsung menulis semua kode sebelum rencana dijelaskan.

Tahap 2 — DATA DAN MODEL
- Implementasikan loader dataset, validasi 303 baris/14 kolom, parsing missing value, pemberian nama kolom, dan target biner.
- Buat pipeline preprocessing: numerik (median imputer + StandardScaler), kategorikal (most_frequent imputer + OneHotEncoder handle_unknown='ignore'), lalu LogisticRegression.
- Pisahkan data secara stratified dengan random_state=42.
- Buat evaluasi dan simpan grafik serta metadata aktual.
- Tambahkan validasi bahwa tidak ada num/target di dalam daftar fitur.

Tahap 3 — APLIKASI WEB
- Buat aplikasi Flask yang memuat pipeline Joblib yang telah dilatih.
- Buat halaman beranda, prediksi, dan tentang model.
- Pakai select/dropdown untuk kategori agar kode kategorinya valid.
- Validasi input di sisi server dan tampilkan pesan kesalahan yang jelas.
- Tampilkan probabilitas kedua kelas dan label “menurut model”.
- Jangan tampilkan prediksi sebagai kepastian medis.

Tahap 4 — PENGUJIAN
- Jalankan training dari awal.
- Jalankan tes loader, target, pipeline, penyimpanan-pemuatan model, dan route Flask.
- Cek bahwa prediksi melalui web menggunakan pipeline yang sama dan urutan fitur yang tepat.
- Perbaiki error yang ditemukan, lalu jalankan kembali pengujian.

TAHAP 5 — DOKUMENTASI
- Buat README.md dengan perintah instalasi, training model, menjalankan Flask, struktur folder, definisi fitur, transformasi target, metrik aktual, dan keterbatasan dataset.
- Pastikan requirements.txt lengkap.

DESAIN:
Buat UI sederhana, modern, bersih, responsif, dan mudah dipakai. Gunakan bahasa Indonesia. Hindari desain berlebihan. Form harus rapi, label input jelas, dan hasil prediksi mudah dibaca.

OUTPUT AKHIR:
1. Source code lengkap.
2. Model pipeline Joblib dan metadata.
3. Grafik evaluasi.
4. Aplikasi Flask yang berjalan lokal.
5. README dan requirements.txt.
6. Ringkasan file yang dibuat, perintah menjalankan proyek, hasil tes, serta hal-hal yang belum berhasil.

Mulai dari Tahap 1. Jangan mengarang output atau menyatakan proyek berhasil sebelum benar-benar menjalankan dan memeriksa hasilnya.
```

---

## 9. Prompt lanjutan per tahap

Gunakan satu per satu. Jangan meminta AI mengerjakan semua tahap sekaligus bila proyek belum teruji.

### Prompt A — audit dataset

```text
Audit dataset saya sebelum modeling. Baca data/processed.cleveland.data tanpa header dengan 14 nama kolom yang sudah ditentukan dalam HEART_DISEASE_VIBE_CODING.md. Tampilkan shape, lima baris pertama, tipe data, jumlah missing value per kolom, jumlah duplikasi, rentang nilai, distribusi num asli, dan distribusi target biner. Pastikan target dibuat dengan num > 0. Jangan melakukan modeling dahulu. Jika ada ketidaksesuaian, jelaskan dan jangan memperbaiki secara diam-diam.
```

### Prompt B — training dan evaluasi

```text
Implementasikan train_model.py sesuai HEART_DISEASE_VIBE_CODING.md. Gunakan Pipeline dan ColumnTransformer, stratified train/test split, Logistic Regression, serta evaluasi Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC-AUC, Classification Report, dan ROC Curve. Simpan pipeline Joblib, metadata JSON, dan gambar grafik. Pastikan semua preprocessing di-fit hanya pada data training. Jalankan script dan laporkan angka aktual dari output, bukan angka perkiraan.
```

### Prompt C — bangun Flask

```text
Bangun aplikasi Flask sesuai HEART_DISEASE_VIBE_CODING.md. Muat artifacts/heart_disease_pipeline.joblib, jangan melatih model pada setiap request. Buat form untuk 13 fitur input, dropdown untuk fitur kategorikal, validasi server-side, serta halaman hasil yang menampilkan kelas dan probabilitas kedua kelas. Pastikan urutan dan nama fitur sama persis dengan saat training. Tambahkan disclaimer non-diagnostik. Setelah implementasi, jalankan aplikasi dan uji route utama.
```

### Prompt D — review kualitas dan debugging

```text
Lakukan code review menyeluruh terhadap proyek berdasarkan acceptance criteria di HEART_DISEASE_VIBE_CODING.md. Cari bug, data leakage, preprocessing leakage, salah pemetaan kolom, kesalahan encoding, ketidaksesuaian urutan fitur, kesalahan interpretasi probabilitas, kelemahan validasi form, dan error path di Windows. Periksa juga apakah metrik yang ditampilkan berasal dari training run aktual. Perbaiki masalah yang terbukti ada, jalankan ulang semua tes, dan laporkan setiap perubahan serta hasil tes. Jangan mengklaim lolos jika tes belum dijalankan.
```

### Prompt E — persiapan laporan TA

```text
Berdasarkan kode, dataset, dan hasil eksekusi proyek yang benar-benar tersedia, susun bahan laporan TA-3 dengan bahasa Indonesia formal yang mudah dipahami. Buat bagian: latar belakang, rumusan masalah, tujuan, deskripsi dataset, definisi variabel, preprocessing, pembagian data, metode Logistic Regression, evaluasi, implementasi Flask, hasil dan pembahasan, keterbatasan, serta kesimpulan. Gunakan angka metrik aktual dari artifacts/model_metadata.json atau output training. Jangan mengarang hasil, referensi, atau klaim medis. Jelaskan bahwa dataset Cleveland hanya 303 sampel dan aplikasi bukan alat diagnosis.
```

---

## 10. Cara menggunakan dengan VS Code, Copilot, dan Kilo

1. Buat folder proyek, misalnya `heart-disease-logreg`, lalu buka folder tersebut di VS Code.
2. Simpan dokumen ini di root proyek dengan nama `HEART_DISEASE_VIBE_CODING.md`.
3. Ekstrak `heart+disease.zip` dan salin hanya `processed.cleveland.data` ke `data/processed.cleveland.data`.
4. Buka Copilot Chat atau Kilo Code dalam mode Agent.
5. Minta AI membaca `HEART_DISEASE_VIBE_CODING.md`, lalu kirim **Prompt utama**.
6. Tinjau rencana yang diberikan. Minta AI menjelaskan setiap perubahan sebelum menyetujui perubahan besar.
7. Lanjutkan memakai Prompt A sampai E secara bertahap.
8. Jalankan perintah di terminal sendiri dan periksa hasilnya. Jangan menganggap kode berhasil hanya karena AI menyatakan selesai.

Perintah awal yang lazim digunakan pada Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python app.py
```

Jika PowerShell memblokir aktivasi environment, gunakan terminal Command Prompt atau ikuti kebijakan keamanan komputer yang digunakan. Jangan mengubah Execution Policy global hanya demi menjalankan proyek.

Buka alamat lokal yang dicetak oleh Flask di terminal, biasanya `http://127.0.0.1:5000`.

---

## 11. Hal yang perlu diperhatikan saat presentasi

- Logistic Regression digunakan untuk klasifikasi biner, bukan untuk memprediksi diagnosis medis secara pasti.
- Sigmoid mengubah skor linear menjadi nilai probabilitas model pada rentang 0 sampai 1.
- Threshold default umumnya 0,5; jelaskan bahwa perubahan threshold dapat mengubah Precision dan Recall.
- Confusion Matrix menunjukkan TP, TN, FP, dan FN.
- Accuracy tidak boleh dibahas sendirian. Bahas Precision, Recall, F1-Score, dan ROC-AUC.
- Recall kelas positif menunjukkan proporsi data berlabel positif yang berhasil dikenali model.
- Hasil evaluasi hanya menggambarkan performa pada pembagian data dan dataset ini; generalisasi ke populasi lain tidak dapat diasumsikan.
- Dataset memiliki 303 sampel sehingga keterbatasan ukuran data wajib disebutkan.
- Hindari menyimpulkan bahwa fitur tertentu menyebabkan penyakit. Model ini mempelajari pola asosiasi pada data, bukan hubungan sebab-akibat.

---

## 12. Sumber data dan dokumentasi

Dokumentasi utama yang ikut di dalam ZIP adalah `heart-disease.names`. File tersebut menjelaskan asal empat kumpulan data, atribut, arti kode target, distribusi kelas, dan penanda missing value. Gunakan dokumentasi tersebut saat menulis laporan.

Dataset Heart Disease juga tersedia pada UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/45/heart+disease

**Catatan penggunaan dataset:** Sertakan atribusi/sumber dataset pada README dan laporan. Periksa ketentuan penggunaan dataset sebelum mendistribusikan ulang file mentah di repository publik.

---

# 13. Prompt terpadu: optimasi, evaluasi, profesionalisasi repository, dan kesiapan Vercel

Gunakan prompt ini **setelah aplikasi utama sudah dibuat dan dapat dijalankan secara lokal**. Prompt ini menggantikan prompt optimasi sebelumnya dan menyatukan empat sasaran: kepatuhan akademik terhadap Modul 03, validitas eksperimen Logistic Regression, kualitas aplikasi web, serta kesiapan deployment ke Vercel. Jalankan melalui Copilot Chat atau Kilo Code dalam mode Agent.

> **Catatan deployment:** Dokumentasi Vercel saat ini menyediakan dukungan Flask melalui Python Runtime. Vercel dapat mendeteksi objek WSGI `app` pada entry point yang dikenali. Panduan Flask Vercel juga mengarahkan aset statis ke `public/` dan menyarankan `vercel dev` untuk pengujian lokal yang mendekati deployment. Periksa lagi dokumentasi resmi bila konfigurasi runtime berubah.
>
> Dokumentasi: [Flask on Vercel](https://vercel.com/docs/frameworks/backend/flask), [Panduan deploy Flask](https://vercel.com/kb/guide/ship-a-flask-app-on-vercel), [Python Runtime](https://vercel.com/docs/functions/runtimes/python), [Function Limitations](https://vercel.com/docs/functions/limitations).

## 13.1 Prompt utama optimasi dan evaluasi menyeluruh

Salin seluruh prompt berikut ke Copilot atau Kilo:

```text
BERTINDAK SEBAGAI:
Senior Machine Learning Engineer, Senior Python/Flask Developer, Vercel Deployment Engineer, UI/UX Designer, Software Architect, dan QA Engineer.

KONTEKS PROYEK:
Saya sedang menyelesaikan TA-3 Modul 03 Pembelajaran Mesin dengan judul:
"Implementasi Algoritma Logistic Regression untuk Klasifikasi Indikasi Penyakit Jantung Berdasarkan Dataset Cleveland Berbasis Web."

Baca seluruh HEART_DISEASE_VIBE_CODING.md dan periksa seluruh repository sebelum melakukan perubahan. Dokumen tersebut adalah spesifikasi utama proyek. Pertahankan dataset Cleveland, definisi target, dan Logistic Regression sebagai algoritma utama. Target deployment adalah Vercel dengan backend Flask.

TUJUAN AKHIR:
Menghasilkan repository yang rapi, profesional, mudah dipahami, dapat direproduksi, sesuai ketentuan Modul 03, memiliki aplikasi web yang responsif, lulus pengujian, dan siap diuji melalui Preview Deployment di Vercel.
Jangan menyatakan aplikasi sudah live sebelum deployment sungguhan berhasil dan URL-nya diuji.

==================================================
A. AUDIT AWAL DAN PERENCANAAN
==================================================
1. Audit seluruh folder dan file proyek sebelum mengedit.
2. Identifikasi file penting, file duplikat, kode tidak terpakai, dependency berlebih, konfigurasi yang bertentangan, dan file lokal yang tidak boleh masuk repository.
3. Periksa dataset, pipeline/model Joblib, metadata, template, aset frontend, route Flask, tes, README, requirements, dan konfigurasi Vercel.
4. Buat tabel audit berisi: area, kondisi sekarang, bukti file, risiko, dan tindakan perbaikan.
5. Buat rencana perubahan dan daftar file yang akan dibuat, dipindahkan, digabung, atau dihapus beserta alasannya.
6. Jangan menghapus atau menimpa file penting sebelum memastikan fungsinya dan memastikan tidak ada perubahan pengguna yang hilang.
7. Hindari rewrite total. Pertahankan implementasi yang sudah benar dan ubah hanya bagian yang memang perlu.

==================================================
B. KEPATUHAN TERHADAP MODUL 03
==================================================
Pastikan implementasi akademik memuat:
1. Logistic Regression sebagai algoritma klasifikasi biner utama.
2. Penjelasan peran sigmoid dan probabilitas hasil model.
3. Confusion Matrix dengan TN, FP, FN, dan TP pada posisi yang tepat.
4. Accuracy, Precision, Recall, dan F1-Score.
5. ROC Curve dan ROC-AUC yang dihitung dari probabilitas kelas positif.
6. Pemisahan data training dan testing yang benar serta stratified.
7. Preprocessing yang hanya di-fit pada data training melalui Pipeline/ColumnTransformer.
8. Penyimpanan preprocessing dan model sebagai satu pipeline Joblib.
9. Aplikasi web yang menerima input pengguna dan menampilkan prediksi serta probabilitas.
10. Interpretasi hasil dan implikasi kesalahan klasifikasi secara proporsional.

Buat matriks kesesuaian terhadap ketentuan TA-3 di modul. Setiap klaim "terpenuhi" wajib memiliki bukti berupa lokasi kode, output, atau tes yang benar-benar dijalankan.

==================================================
C. VALIDASI DATASET DAN DEFINISI TARGET
==================================================
1. Gunakan hanya data/processed.cleveland.data sebagai dataset utama.
2. Pastikan loader membaca 303 baris dan 14 kolom sesuai dokumentasi proyek.
3. Baca tanda '?' sebagai missing value dan verifikasi lokasi serta jumlahnya.
4. Terapkan nama kolom sesuai urutan yang ditentukan di HEART_DISEASE_VIBE_CODING.md.
5. Bentuk target biner dengan aturan num == 0 menjadi 0 dan num > 0 menjadi 1.
6. Jangan memasukkan kolom num maupun target ke daftar fitur.
7. Laporkan jumlah baris, missing values, duplikasi, distribusi target, dan rentang nilai aktual.
8. Jangan membuat data sintetis, mengubah isi file sumber, membuang sampel secara sembarangan, atau mengganti dataset.
9. Bila ditemukan hasil yang berbeda dari pemeriksaan awal, tampilkan temuan aktual dan jelaskan penyebabnya; jangan menyamarkan perbedaan.

==================================================
D. OPTIMASI MODEL SECARA ILMIAH DAN DAPAT DIREPRODUKSI
==================================================
1. Pertahankan Logistic Regression sebagai model utama.
2. Buat baseline yang reproducible dengan random_state=42 dan stratified train/test split.
3. Gunakan Pipeline dan ColumnTransformer. Fitur numerik memakai median imputer dan StandardScaler. Fitur kategorikal memakai most_frequent imputer dan OneHotEncoder(handle_unknown='ignore'), sesuai spesifikasi proyek.
4. Jangan melakukan imputasi, scaling, encoding, seleksi fitur, atau resampling dengan mempelajari seluruh dataset sebelum split.
5. Jika tuning diperlukan, lakukan GridSearchCV/RandomizedSearchCV menggunakan StratifiedKFold pada data training saja. Jelaskan ruang parameter, scoring, dan alasan pemilihannya.
6. Bandingkan baseline dan hasil tuning pada fold validasi yang sama. Laporkan mean dan standard deviation cross-validation.
7. Test set hanya digunakan untuk evaluasi final setelah konfigurasi dipilih. Jangan menggunakan test set untuk memilih hyperparameter, threshold, fitur, atau preprocessing.
8. Gunakan threshold 0.5 sebagai hasil utama. Eksperimen threshold lain hanya boleh berdasarkan prediksi validation/out-of-fold pada training data dan dilaporkan terpisah.
9. Jangan memaksakan SMOTE. Bila diuji sebagai eksperimen tambahan, tempatkan resampling di dalam pipeline imbalanced-learn dan hanya pada fold training.
10. Hitung Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report, ROC Curve, dan ROC-AUC pada kelas positif target=1.
11. Periksa urutan label Confusion Matrix agar TN, FP, FN, TP tidak salah interpretasi.
12. Bandingkan baseline dan tuned model secara jujur. Jangan menjanjikan akurasi yang lebih tinggi atau mengubah pembagian data demi skor yang bagus.
13. Simpan parameter, seed, versi library, hasil CV, metrik test, threshold, dan timestamp training ke metadata JSON.
14. Pastikan dashboard dan laporan membaca hasil dari training run aktual, bukan angka hard-coded.
15. Jangan mengklaim probabilitas model sebagai probabilitas klinis tervalidasi.

==================================================
E. PROFESIONALISASI DESAIN DAN PENGALAMAN PENGGUNA
==================================================
1. Buat desain modern, bersih, minimalis, konsisten, dan sesuai aplikasi akademik/data science. Hindari tampilan generik, dekorasi berlebihan, gradient berlebihan, dan animasi berat.
2. Gunakan palet warna profesional yang terbatas, kontras teks memadai, tipografi mudah dibaca, jarak dan ukuran komponen konsisten, serta hierarki heading yang jelas.
3. Pastikan layout responsif untuk desktop, tablet, dan smartphone.
4. Sediakan navigasi yang jelas serta header/footer yang konsisten.
5. Beranda menjelaskan masalah, tujuan, dataset Cleveland, Logistic Regression, dan tautan menuju halaman prediksi serta evaluasi.
6. Form prediksi mengelompokkan 13 fitur secara logis, memberi label dan satuan yang tepat, menggunakan select/radio untuk fitur kategorikal, dan memberi petunjuk yang jelas.
7. Validasi wajib dilakukan di server. Validasi HTML/JavaScript hanya tambahan.
8. Hasil prediksi menampilkan label "Ada indikasi menurut model" atau "Tidak ada indikasi menurut model", probabilitas kelas 0 dan kelas 1, serta penjelasan singkat tentang arti probabilitas.
9. Tampilkan disclaimer dengan jelas: aplikasi adalah prototipe pembelajaran, bukan alat diagnosis, bukan saran pengobatan, dan bukan pengganti tenaga medis.
10. Halaman evaluasi menampilkan nilai metrik aktual, definisi metrik, Confusion Matrix, ROC Curve, distribusi target, dan informasi dataset jika artefaknya tersedia.
11. Jangan menampilkan grafik kosong, angka contoh, metrik palsu, tombol mati, link rusak, atau placeholder yang tertinggal.
12. Terapkan aksesibilitas dasar: label input terkait, navigasi keyboard, focus state terlihat, alt text untuk gambar, dan pesan error yang jelas.
13. Pastikan tidak ada horizontal overflow pada layar kecil.

==================================================
F. STRUKTUR FILE DAN KEBERSIHAN REPOSITORY
==================================================
Rapikan repository dengan standar proyek Python yang profesional dan mudah dipelihara.

Struktur target boleh disesuaikan dengan kondisi aktual, tetapi upayakan pemisahan yang jelas seperti:

heart-disease-logreg/
  app.py                         # entry point Flask yang dikenali Vercel
  train_model.py                 # training terpisah, tidak dijalankan saat request
  requirements.txt               # dependency runtime yang teruji dan dipin
  .python-version                # versi Python yang didukung dan sudah diuji
  .gitignore
  .vercelignore                  # hanya bila dibutuhkan dan valid untuk deployment
  vercel.json                    # hanya bila ada kebutuhan konfigurasi eksplisit
  README.md
  HEART_DISEASE_VIBE_CODING.md
  data/
    processed.cleveland.data     # dataset sumber; jangan dimasukkan ke bundle runtime bila tidak diperlukan
  artifacts/
    heart_disease_pipeline.joblib
    model_metadata.json
  src/
    __init__.py
    data_loader.py
    preprocessing.py
    evaluation.py
  templates/
    base.html
    index.html
    predict.html
    about.html
  public/
    css/
      style.css
    js/
    images/
  reports/
    confusion_matrix.png
    roc_curve.png
    target_distribution.png
  tests/
    test_data.py
    test_model.py
    test_app.py

ATURAN KERAPIAN:
1. Jangan menaruh semua kode dalam satu file jika pemisahan modul membuatnya lebih jelas.
2. Hindari file duplikat seperti app_old.py, final2.py, style-new.css, backup, atau file eksperimen yang tidak dipakai. Jangan hapus tanpa audit.
3. Pisahkan dependency runtime dan development bila memang diperlukan. Jangan memasukkan paket yang tidak digunakan.
4. Pin versi dependency yang telah diuji, terutama scikit-learn agar kompatibel dengan file Joblib yang disimpan.
5. Gunakan pathlib dan path absolut berbasis lokasi file untuk membaca artifacts; jangan mengandalkan current working directory.
6. Buat .gitignore untuk .venv/, venv/, __pycache__/, *.pyc, .env, .env.*, .pytest_cache/, .coverage, .idea/, .vscode/ bila sesuai, notebook checkpoint, OS files, logs, dan file sementara.
7. Jangan meng-ignore artifacts/heart_disease_pipeline.joblib atau metadata yang memang diperlukan aplikasi saat deployment.
8. Dataset mentah, notebook, cache, tes, dan laporan pengembangan tidak perlu masuk production bundle bila tidak diperlukan runtime. Pertahankan file sumber di repository kerja sesuai kebijakan distribusi dataset.
9. Jangan commit secret, token, kredensial, file .env, atau file konfigurasi lokal yang sensitif.
10. Tambahkan komentar hanya pada logika yang memang perlu dijelaskan. Gunakan nama fungsi, variabel, dan file yang konsisten.
11. README harus menjelaskan tujuan, dataset dan atribusi, struktur proyek, instalasi Windows, training, menjalankan Flask, menjalankan tes, evaluasi, keterbatasan, dan deployment Vercel.
12. Buat atau perbarui dokumentasi singkat untuk file penting agar pengguna lain dapat memahami proyek tanpa membaca seluruh source code.
13. Pastikan repository hanya berisi file yang relevan, tidak ada hasil build atau artefak sementara yang tidak diperlukan.

==================================================
G. KOMPATIBILITAS DAN KESIAPAN DEPLOYMENT VERCEL
==================================================
Target: deploy aplikasi Flask pada Vercel Python Runtime.

1. Pastikan Flask app diekspor sebagai objek WSGI tingkat atas bernama `app` di entry point yang dikenali Vercel, misalnya app.py di root proyek.
2. Letakkan `app.run()` hanya di dalam `if __name__ == '__main__':` untuk development lokal. Jangan menjalankan development server saat modul di-import oleh Vercel.
3. Jangan jalankan training model pada saat cold start atau setiap request. Training dilakukan dengan train_model.py secara terpisah; runtime hanya memuat artifacts yang sudah disiapkan.
4. Muat pipeline Joblib dengan path yang dibangun dari `Path(__file__).resolve()` dan pastikan artifacts ikut ter-bundle dalam deployment.
5. Pin versi Python dan dependency yang kompatibel. Uji bahwa scikit-learn yang digunakan untuk load model kompatibel dengan versi yang digunakan saat menyimpan model.
6. Ikuti panduan Flask Vercel untuk aset statis: gunakan folder `public/` untuk aset yang disajikan melalui CDN dan perbaiki referensi URL pada template. Jangan mengandalkan Flask `static_folder` lokal tanpa menguji perilakunya di Vercel.
7. Pastikan semua template Jinja yang digunakan ikut dalam deployment bundle. Uji beranda dan seluruh route setelah build/deploy.
8. Perlakukan filesystem deployment sebagai ephemeral/read-only untuk kebutuhan aplikasi. Jangan menyimpan prediksi, upload, model hasil training, log, atau data pengguna ke folder source saat request. Gunakan `/tmp` hanya untuk file sementara bila benar-benar diperlukan.
9. Jangan mengandalkan SQLite lokal atau penyimpanan lokal persisten. Jangan menambahkan database atau layanan eksternal tanpa alasan dan persetujuan.
10. Periksa ukuran deployment bundle dan hapus dependency development, dataset, notebook, cache, dan tes dari bundle runtime jika tidak dibutuhkan. Jangan sampai bundle Python mendekati batas platform.
11. Pastikan tidak ada secret atau environment file di repository. Buat environment variable hanya bila aplikasi benar-benar membutuhkannya. Jangan meminta saya mengirim VERCEL_TOKEN melalui chat.
12. Jangan menambahkan `vercel.json` jika zero-config Flask detection sudah cukup. Jika perlu, buat konfigurasi minimal berdasarkan dokumentasi resmi terkini dan jelaskan alasannya. Jangan menyalin konfigurasi lama yang tidak relevan.
13. Tambahkan route health check sederhana seperti `/health` yang tidak membocorkan informasi sensitif dan dapat digunakan untuk smoke test.
14. Uji dengan Vercel CLI melalui `vercel dev` bila CLI tersedia. Jangan hanya mengandalkan `flask run`.
15. Pastikan GET halaman utama, GET halaman form, POST prediksi, route evaluasi, route health check, dan seluruh static assets berjalan.
16. Preview Deployment harus diuji terlebih dahulu. Jangan menyatakan siap production hanya karena build lokal berhasil.
17. Jangan menjalankan deploy production tanpa permintaan eksplisit. Dalam tugas optimasi ini, siapkan proyek dan berikan instruksi deploy; deployment aktual memerlukan akun dan tindakan pengguna.
18. Laporkan status dengan istilah yang tepat: `belum siap`, `siap diuji lokal`, `siap Preview Deployment`, `Preview berhasil diuji`, atau `Production live`. Gunakan status terakhir hanya jika benar-benar terbukti.

==================================================
H. PENGUJIAN WAJIB DAN QUALITY GATE
==================================================
Buat atau perbarui automated tests untuk memeriksa:
- Loader membaca 303 baris dan 14 kolom.
- Missing value '?' terdeteksi pada kolom yang benar.
- Target biner sesuai aturan num == 0 -> 0 dan num > 0 -> 1.
- num dan target tidak masuk ke fitur.
- train/test split stratified dan kedua kelas terwakili.
- Pipeline preprocessing dan Logistic Regression berhasil fit.
- predict dan predict_proba berjalan.
- Probabilitas tiap kelas berada pada rentang 0 sampai 1 dan jumlahnya mendekati 1.
- Metrik menggunakan kelas positif target=1 dan Confusion Matrix dipetakan benar.
- Model dapat disimpan dan dimuat kembali dengan prediksi konsisten.
- Semua 13 fitur diterima dengan urutan dan nama yang tepat.
- Form menolak input kosong, tidak valid, NaN/infinity, kategori tidak dikenal, dan nilai di luar rentang teknis yang ditentukan.
- Route Flask utama merespons benar.
- Aset CSS/JS/gambar tidak 404.
- Model artifacts dapat dimuat dari path deployment.
- Tidak ada training yang terjadi pada request aplikasi.
- Tampilan tidak overflow pada ukuran layar mobile.
- `vercel dev` dapat menjalankan aplikasi jika Vercel CLI tersedia.

Jalankan tes yang tersedia dan catat command serta hasil aktual. Jangan menandai tes lulus bila belum dijalankan. Bila ada tes yang tidak dapat dijalankan, jelaskan alasan dan tandai belum diverifikasi.

==================================================
I. URUTAN IMPLEMENTASI
==================================================
Kerjakan secara bertahap dan setelah setiap tahap jalankan tes terkait:
1. Audit repository dan tampilkan rencana.
2. Rapikan struktur dan konfigurasi tanpa menghilangkan fungsi yang ada.
3. Audit data, pipeline, baseline, tuning, dan evaluasi.
4. Rapikan backend Flask dan integrasi artifacts.
5. Poles UI/UX dan aksesibilitas.
6. Adaptasi entry point, aset, dependency, dan konfigurasi untuk Vercel.
7. Jalankan unit test, integration test, dan smoke test.
8. Perbarui README dan dokumentasi.
9. Buat laporan hasil akhir dan daftar blocker.

Jangan melakukan perubahan besar sekaligus. Setelah tiap tahap, ringkas file yang berubah dan alasan perubahannya.

==================================================
J. OUTPUT AKHIR YANG WAJIB DIBERIKAN
==================================================
Berikan laporan akhir yang memuat:
1. Status kepatuhan terhadap Modul 03 beserta bukti.
2. Audit struktur repository dan daftar file yang dibuat, dipindahkan, diubah, atau dihapus.
3. Daftar perbaikan UI/UX yang benar-benar dilakukan.
4. Perbandingan baseline dan tuned Logistic Regression berdasarkan CV serta test set yang terpisah dengan benar.
5. Metrik final aktual: Accuracy, Precision, Recall, F1-Score, ROC-AUC, dan Confusion Matrix.
6. Hasil unit test, integration test, dan smoke test beserta command yang dijalankan.
7. Status kompatibilitas Vercel dan hasil `vercel dev` bila dijalankan.
8. Daftar konfigurasi Vercel yang masih perlu dilakukan secara manual oleh pemilik akun.
9. Instruksi menjalankan proyek dari awal pada Windows.
10. Instruksi deployment Preview dan Production, dengan Preview diuji terlebih dahulu.
11. Keterbatasan dataset dan aplikasi yang masih ada.
12. Daftar blocker yang belum selesai; jangan menyembunyikan kegagalan.

ATURAN MUTLAK:
- Jangan mengganti dataset Cleveland.
- Jangan mengganti Logistic Regression sebagai algoritma utama.
- Jangan melakukan data leakage.
- Jangan mengarang metrik, hasil test, screenshot, URL, atau status deployment.
- Jangan menyimpan kredensial dalam source code atau meminta saya mengirim secret melalui chat.
- Jangan menghapus fitur yang berjalan tanpa alasan.
- Jangan menyatakan proyek selesai, profesional, sesuai modul, atau siap production sebelum quality gate yang relevan lulus.

Mulai dengan audit repository dan rencana perubahan. Setelah itu implementasikan secara bertahap, uji, dan dokumentasikan hasilnya.
```

## 13.2 Prompt khusus audit kebersihan file dan struktur proyek

Gunakan prompt ini bila repository mulai berantakan atau setelah beberapa kali iterasi Vibe Coding:

```text
Lakukan repository cleanup terhadap proyek Heart Disease Prediction.

Sebelum mengubah apa pun, audit seluruh file dan identifikasi:
- file duplikat atau versi lama;
- file eksperimen yang tidak digunakan;
- import dan dependency yang tidak terpakai;
- cache, log, file sementara, dan output build;
- konfigurasi yang saling bertentangan;
- file yang wajib ada untuk aplikasi Flask dan deployment Vercel;
- artifacts model yang diperlukan runtime.

Rapikan struktur proyek secara profesional. Pisahkan source code, data sumber, artifacts model, template, public assets, laporan, dan tests. Pertahankan entry point Flask yang dapat dikenali Vercel. Pastikan model Joblib dan metadata yang dibutuhkan runtime tidak terhapus atau ter-ignore.

Buat/perbarui .gitignore dan requirements.txt. Jangan memasukkan .env, token, .venv, __pycache__, cache, notebook checkpoints, file OS, atau file development yang tidak diperlukan ke repository/deployment. Jangan menghapus dataset sumber dari working directory. Dataset mentah, notebook, dan tests tidak perlu ikut production bundle bila tidak diperlukan runtime.

Pindahkan aset frontend ke public/ bila dibutuhkan oleh konfigurasi Flask di Vercel, lalu perbarui semua URL di template dan uji satu per satu agar tidak ada 404. Jangan membuat konfigurasi Vercel yang tidak diperlukan.

Setelah cleanup:
1. Tampilkan tree folder terbaru.
2. Tampilkan daftar file yang dipindah, digabung, diubah, dan dihapus beserta alasannya.
3. Pastikan aplikasi tetap berjalan lokal.
4. Jalankan semua tests.
5. Jalankan vercel dev bila tersedia.
6. Periksa ulang bahwa semua file runtime yang dibutuhkan deployment ikut ter-bundle.
7. Perbarui README dengan struktur proyek final.

Jangan menyatakan cleanup berhasil sebelum aplikasi dan tes tetap lulus.
```

## 13.3 Prompt evaluasi akhir sebagai simulasi dosen penguji dan reviewer deployment

```text
Lakukan FINAL REVIEW terhadap proyek TA-3 Heart Disease Prediction sebagai dosen penguji Modul 03 sekaligus reviewer software/deployment.

Jangan langsung mengubah kode. Periksa dan buat laporan audit terlebih dahulu berdasarkan bukti repository dan hasil command yang benar-benar dijalankan.

BAGIAN AKADEMIK:
1. Validasi dataset Cleveland, 303 baris, 14 kolom, missing values, dan definisi target biner.
2. Pastikan num dan target tidak masuk sebagai fitur.
3. Audit train/test split, stratifikasi, Pipeline, imputasi, encoding, scaling, dan potensi data leakage.
4. Pastikan Logistic Regression merupakan model utama.
5. Verifikasi baseline, tuning, cross-validation, dan test evaluation dilakukan pada subset yang tepat.
6. Verifikasi Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve, ROC-AUC, dan interpretasinya.
7. Periksa apakah angka yang ditulis pada README/dashboard sama dengan output training aktual.
8. Periksa interpretasi TN, FP, FN, TP dan trade-off Precision/Recall.

BAGIAN APLIKASI:
1. Uji seluruh halaman, navigasi, form, validasi, hasil prediksi, dan probabilitas.
2. Pastikan model Joblib yang digunakan aplikasi sama dengan pipeline final.
3. Periksa 13 fitur, nama kolom, tipe data, dan urutannya.
4. Pastikan disclaimer non-diagnostik tersedia.
5. Nilai profesionalitas UI: konsistensi, responsivitas, aksesibilitas dasar, dan tidak ada link/tombol mati.

BAGIAN REPOSITORY:
1. Periksa struktur direktori, penamaan file, README, requirements, .gitignore, dan file konfigurasi.
2. Cari file duplikat, file tidak terpakai, hard-coded metrics, secret, cache, dan artefak sementara.
3. Pastikan artifacts yang dibutuhkan runtime tidak terhapus dan file development tidak membebani deployment.

BAGIAN VERCEL:
1. Verifikasi Flask entry point mengekspor objek `app` yang dapat dideteksi Vercel.
2. Verifikasi Python version dan dependency sesuai runtime.
3. Pastikan asset publik di folder public/ termuat.
4. Pastikan model Joblib, metadata, dan templates dapat diakses di deployment bundle.
5. Pastikan tidak ada training saat request dan tidak ada ketergantungan pada penyimpanan lokal persisten.
6. Periksa path, ukuran bundle, route, health check, logging, serta error handling.
7. Jalankan vercel dev bila tersedia. Jika tidak tersedia, jelaskan bahwa kompatibilitas runtime belum diuji.
8. Bedakan status siap diuji deploy, Preview berhasil diuji, dan Production live. Jangan menyatakan Production live tanpa URL dan tes nyata.

FORMAT HASIL:
Buat tabel dengan kolom: komponen, status (LULUS/PERLU PERBAIKAN/BELUM DIVERIFIKASI), bukti, risiko, dan rekomendasi.

Setelah audit, perbaiki semua masalah yang aman untuk diperbaiki. Jangan menunggu persetujuan untuk perbaikan rutin yang jelas dan tidak merusak. Jangan melakukan perubahan destruktif atau menghapus data pengguna.

Jalankan ulang tes setelah perbaikan. Akhiri dengan:
- ringkasan kepatuhan Modul 03;
- metrik aktual;
- ringkasan hasil tes;
- struktur folder final;
- status kesiapan Vercel;
- langkah manual yang masih harus saya lakukan;
- blocker yang belum terselesaikan.

Dilarang mengarang angka, hasil tes, screenshot, URL deployment, atau status berhasil.
```

## 13.4 Quality gate sebelum menyebut proyek siap deploy

Proyek hanya boleh disebut **siap Preview Deployment** bila seluruh hal berikut sudah dibuktikan:
- [ ] Audit Modul 03 selesai dan tidak ada pelanggaran metodologi yang diketahui.
- [ ] Baseline dan model hasil tuning dibandingkan tanpa kebocoran test data.
- [ ] Metrik final dihitung dari test set yang ditahan.
- [ ] Model dan preprocessing tersimpan dalam pipeline Joblib yang sama.
- [ ] Flask mengekspor objek WSGI tingkat atas bernama `app`.
- [ ] Tidak ada training model pada request/cold start.
- [ ] Path model, metadata, dan template dapat diakses dari entry point deployment.
- [ ] Aset frontend yang dibutuhkan ditempatkan di `public/` dan tidak 404.
- [ ] Versi Python dan dependency kompatibel serta sudah diuji.
- [ ] Tidak ada secret, virtual environment, cache, atau file lokal sensitif di repository.
- [ ] README, requirements, .gitignore, dan struktur folder sudah rapi.
- [ ] Semua automated tests lulus, atau kegagalan yang diketahui didokumentasikan.
- [ ] `vercel dev` berhasil dijalankan dan smoke test route utama lulus.

Preview deployment yang sesungguhnya tetap harus diuji melalui URL yang diberikan Vercel. Jika belum dilakukan, tulis **"siap diuji Preview Deployment"**, bukan **"sudah berhasil deploy"**.

---

## 14. Checklist evaluasi akhir sebelum dikumpulkan

### Kesesuaian metode dan modul
- [ ] Algoritma utama yang digunakan tetap Logistic Regression.
- [ ] Target biner mengikuti aturan `num == 0` menjadi 0 dan `num > 0` menjadi 1.
- [ ] Tidak ada target leakage atau preprocessing leakage.
- [ ] Baseline dan tuning dibandingkan dengan prosedur validasi yang benar.
- [ ] Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC Curve, dan ROC-AUC tersedia.
- [ ] Interpretasi FP dan FN sesuai posisi pada Confusion Matrix.
- [ ] Hasil model tidak diklaim sebagai diagnosis medis.

### Kualitas aplikasi
- [ ] Semua navigasi dan tombol berfungsi.
- [ ] Form memiliki 13 fitur dengan jenis input yang sesuai.
- [ ] Server-side validation berjalan.
- [ ] Probabilitas ditampilkan dari `predict_proba`, bukan angka statis.
- [ ] UI responsif dan dapat digunakan pada ponsel.
- [ ] Warna, tipografi, jarak, tabel, dan grafik konsisten.
- [ ] Disclaimer terlihat pada alur prediksi.
- [ ] Halaman evaluasi memakai metrik dan grafik dari training run aktual.

### Pengujian dan dokumentasi
- [ ] Tes preprocessing, model, serialisasi, route, dan validasi input dijalankan.
- [ ] Training dan aplikasi dapat dijalankan ulang dari petunjuk README di Windows.
- [ ] `requirements.txt` sesuai dependensi yang benar-benar digunakan.
- [ ] Tidak ada file sensitif atau environment lokal dalam repository.
- [ ] README mencantumkan sumber dataset, definisi fitur/target, hasil aktual, dan keterbatasan.

---

# 15. Persiapan Deployment ke Vercel

Bagian ini menambahkan target deployment Vercel tanpa mengubah ruang lingkup akademik TA-3. Aplikasi tetap menggunakan Flask dan Logistic Regression. Vercel menjalankan aplikasi Flask sebagai Python Function; Flask dapat terdeteksi otomatis dari struktur proyek dan objek WSGI bernama `app` pada entry point yang dikenali. Periksa dokumentasi resmi sebelum deployment karena dukungan runtime, batas fungsi, dan konfigurasi dapat berubah.

Dokumentasi resmi yang menjadi acuan:
- Flask di Vercel: https://vercel.com/docs/frameworks/backend/flask
- Panduan deploy Flask: https://vercel.com/kb/guide/ship-a-flask-app-on-vercel
- Python Runtime: https://vercel.com/docs/functions/runtimes/python
- Konfigurasi `vercel.json`: https://vercel.com/docs/project-configuration/vercel-json

## 15.1 Target arsitektur deployment

Gunakan pendekatan sederhana berikut:

- `app.py` di root proyek sebagai entry point Flask dan mengekspor objek tingkat atas bernama `app`.
- Flask menangani halaman, form, dan route prediksi.
- `artifacts/` berisi pipeline Logistic Regression final dan metadata yang diperlukan aplikasi saat runtime.
- `public/` berisi aset statis yang perlu disajikan melalui Vercel CDN, misalnya CSS, JavaScript, ikon, dan gambar. Sesuaikan referensi aset pada template menjadi path publik yang benar.
- `templates/` berisi template HTML/Jinja.
- `requirements.txt` berisi hanya dependency runtime yang diperlukan aplikasi.
- Notebook, dataset mentah, file training sementara, dan artefak eksperimen tidak ikut dibundel bila tidak diperlukan saat aplikasi melayani request.

Struktur target yang disarankan:

```text
heart-disease-prediction/
├── app.py
├── requirements.txt
├── .python-version
├── README.md
├── artifacts/
│   ├── heart_disease_pipeline.joblib
│   └── model_metadata.json
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   └── evaluation.html
├── public/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── images/
├── src/
│   ├── preprocessing.py
│   └── model_inference.py
├── train_model.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_routes.py
└── data/
    └── README.md
```

Struktur di atas adalah rancangan, bukan kewajiban untuk memindahkan semua kode apabila struktur proyek yang sudah ada lebih baik. Jangan menggandakan file atau memecah kode tanpa alasan teknis.

## 15.2 Ketentuan teknis khusus Vercel

1. **Entry point Flask:** pastikan `app.py` berada di root dan membuat objek tingkat atas `app = Flask(__name__)`. Jangan hanya membuat aplikasi di dalam fungsi factory tanpa mengekspor objek yang dikenali Vercel, kecuali entry point dikonfigurasi secara resmi dan diuji.
2. **Jangan menjalankan server development pada import:** `app.run()` hanya boleh berada di dalam `if __name__ == "__main__":` untuk kebutuhan lokal. Vercel mengelola proses request.
3. **Model sudah dilatih sebelum deployment:** training tidak dijalankan saat aplikasi mulai atau saat setiap request. Simpan pipeline final dengan Joblib, sertakan artefak model tersebut dalam repository/deployment bundle, lalu muat sekali saat aplikasi diinisialisasi.
4. **Pipeline preprocessing harus satu paket:** simpan encoder, imputer, scaler (bila digunakan), urutan fitur, dan Logistic Regression di dalam satu `Pipeline`/`ColumnTransformer` yang sesuai. Jangan melakukan transformasi berbeda antara training dan prediksi.
5. **Path file harus absolut berdasarkan lokasi proyek:** jangan mengandalkan current working directory. Gunakan `pathlib.Path(__file__).resolve().parent` atau path helper yang konsisten untuk menemukan model dan template.
6. **Filesystem runtime:** perlakukan filesystem deployment sebagai read-only. Jangan menulis log, model baru, CSV hasil prediksi, cache permanen, atau file upload ke folder proyek saat request. Bila scratch space sementara benar-benar diperlukan, gunakan `/tmp` dan jangan menganggap isinya persisten antar-request. Untuk proyek TA ini, sebaiknya tidak perlu menulis file pada saat inferensi.
7. **Aset statis:** gunakan struktur `public/` yang sesuai panduan Flask Vercel terkini; pastikan HTML mengakses CSS, JS, ikon, dan gambar dengan URL publik yang benar. Jangan mengandalkan folder lokal yang tidak ikut dibundel.
8. **Dependency:** `requirements.txt` harus memuat Flask, scikit-learn, pandas/numpy hanya jika diperlukan saat runtime, joblib, dan paket lain yang benar-benar digunakan. Pin versi yang sudah diuji. Versi scikit-learn saat training dan runtime harus kompatibel, idealnya sama.
9. **Python version:** tentukan versi Python yang didukung Vercel dan kompatibel dengan versi dependency proyek melalui `.python-version` atau konfigurasi proyek yang didukung. Jangan mengunci versi tanpa menguji instalasi dependency di versi tersebut.
10. **Bundle size dan cold start:** ukur ukuran bundle, waktu import, dan waktu load model. Hilangkan dataset mentah, notebook, cache, dan dependency development dari bundle runtime jika tidak dibutuhkan.
11. **Environment variables:** jangan menyimpan secret di source code. Tambahkan `.env` ke `.gitignore`. Proyek prediksi lokal tanpa API eksternal mungkin tidak memerlukan environment variable khusus; jangan membuat variabel rahasia palsu hanya untuk memenuhi deployment.
12. **Keamanan input:** validasi seluruh input di server, tolak nilai tidak numerik, NaN/infinity, serta rentang yang tidak masuk akal. Jangan mengandalkan validasi HTML saja.
13. **Observability:** gunakan logging standar ke stdout/stderr tanpa mencatat data kesehatan sensitif yang dimasukkan pengguna. Jangan menyimpan identitas atau data input pengguna bila tidak diperlukan.
14. **Tidak perlu `vercel.json` jika zero-config Flask detection sudah bekerja.** Tambahkan `vercel.json` hanya bila ada kebutuhan konfigurasi yang jelas dan valid menurut dokumentasi terkini. Hindari konfigurasi `builds` lama atau rewrite yang tidak diperlukan.
15. **Testing deployment:** gunakan `vercel dev` untuk memeriksa perilaku lokal yang mendekati lingkungan Vercel. Jangan hanya mengandalkan `flask run`.

## 15.3 Prompt implementasi Vercel untuk Copilot/Kilo

Salin prompt berikut setelah aplikasi lokal, training, dan evaluasi sudah selesai. Minta Agent membaca keseluruhan repository sebelum mengubah file.

```text
Saya ingin menyiapkan proyek Heart Disease Prediction ini agar dapat di-deploy dan berjalan dengan benar di Vercel.

Baca HEART_DISEASE_VIBE_CODING.md, terutama bagian dataset, preprocessing, training, evaluasi, dan bagian 15 tentang Vercel. Audit seluruh repository sebelum melakukan perubahan. Pertahankan dataset Cleveland dan Logistic Regression sebagai algoritma utama sesuai Modul 03.

TARGET DEPLOYMENT:
- Platform: Vercel.
- Backend: Flask menggunakan Python Runtime Vercel.
- Entry point: app.py di root proyek yang mengekspor objek WSGI tingkat atas bernama `app`.
- Model: pipeline final yang sudah dilatih dan disimpan dengan Joblib.
- Frontend: HTML/Jinja, CSS, dan JavaScript yang sudah ada.
- Static assets: sesuaikan dengan mekanisme folder public/ pada panduan Flask Vercel terkini.

KERJAKAN TAHAP BERIKUT:

1. Audit kompatibilitas
   - Periksa struktur folder, app Flask, route, template, CSS/JS, model Joblib, dan requirements.
   - Identifikasi penggunaan filesystem, path relatif, file yang ditulis saat request, import yang berat, serta dependency yang tidak kompatibel.
   - Jangan mengubah algoritma, definisi target, atau fitur tanpa alasan dan dokumentasi.

2. Siapkan Flask entry point
   - Pastikan Vercel dapat mendeteksi objek `app` pada entry point yang dikenali.
   - `app.run()` hanya untuk eksekusi lokal di dalam blok `if __name__ == "__main__":`.
   - Pastikan semua route halaman utama, prediksi, evaluasi, health check, dan error handler berfungsi.
   - Jangan membuat server atau proses background yang harus hidup terus-menerus.

3. Siapkan model inference
   - Training tetap dijalankan secara terpisah, bukan ketika deployment request berjalan.
   - Muat pipeline Joblib satu kali saat aplikasi diinisialisasi, dengan path berbasis `Path(__file__).resolve()`.
   - Pipeline harus mencakup preprocessing dan Logistic Regression agar fitur diproses konsisten.
   - Periksa kompatibilitas versi scikit-learn antara training dan deployment.
   - Jangan menyimpan ulang model pada setiap request.
   - Tangani kegagalan model file dengan pesan error server yang aman dan log teknis yang tidak membocorkan data input.

4. Rapikan aset dan bundle
   - Pindahkan atau sesuaikan aset publik ke folder `public/` sesuai panduan Vercel.
   - Perbaiki URL aset pada template dan uji apakah seluruh CSS, JS, ikon, dan gambar termuat.
   - Buat requirements.txt yang minimal dan pin versi yang telah diuji.
   - Tentukan versi Python yang kompatibel dan didukung Vercel.
   - Jangan masukkan notebook, virtual environment, cache, dataset mentah, atau test fixtures ke bundle production jika tidak dibutuhkan.
   - Jangan menghapus dataset dari repository kerja; cukup cegah file yang tidak dibutuhkan runtime ikut masuk bundle.

5. Periksa filesystem dan konfigurasi
   - Anggap filesystem deployment read-only.
   - Jangan menulis file hasil prediksi, log, model, atau upload ke direktori source saat request.
   - Gunakan /tmp hanya untuk file sementara yang benar-benar dibutuhkan, bukan penyimpanan permanen.
   - Jangan menambahkan database atau layanan eksternal tanpa kebutuhan.
   - Jangan membuat vercel.json jika auto-detection sudah cukup. Jika diperlukan, buat konfigurasi minimal sesuai dokumentasi Vercel saat ini dan jelaskan setiap properti.
   - Tambahkan .env, .venv, __pycache__, file lokal, dan secret ke .gitignore.

6. Uji lokal dengan cara yang mendekati Vercel
   - Install dependency dari requirements.txt pada environment bersih.
   - Jalankan vercel dev jika Vercel CLI tersedia.
   - Uji GET halaman utama, halaman prediksi, halaman evaluasi, health check, dan POST prediksi.
   - Uji input valid, input kosong, input non-numerik, nilai di luar rentang, NaN, dan infinity.
   - Uji pemuatan model serta satu prediksi contoh.
   - Uji aset statis dan pastikan tidak ada 404.
   - Jalankan seluruh automated tests.

7. Dokumentasi deployment
   - Perbarui README dengan instruksi Windows dan deployment Vercel.
   - Jelaskan cara push repository ke GitHub dan import repository di Vercel.
   - Jelaskan pengaturan Root Directory, framework detection, Python version, build settings, dan environment variables hanya jika benar-benar diperlukan.
   - Berikan perintah CLI preview `vercel` dan production `vercel --prod`.
   - Jangan meminta saya menuliskan VERCEL_TOKEN atau secret ke chat atau source code.
   - Catat batasan dan kemungkinan perbedaan konfigurasi berdasarkan akun/plan Vercel.

8. Laporan hasil
   - Buat daftar file yang dibuat/diubah.
   - Laporkan hasil test dan perintah yang benar-benar dijalankan.
   - Tunjukkan status siap preview atau belum siap beserta blocker yang tersisa.
   - Jangan mengklaim deployment berhasil sebelum URL deployment benar-benar tersedia dan diuji.

ATURAN WAJIB:
- Jangan mengganti dataset Cleveland.
- Jangan mengganti Logistic Regression sebagai model utama.
- Jangan melakukan training ulang secara otomatis pada runtime Vercel.
- Jangan menyimpan kredensial di repository.
- Jangan mengarang URL, environment variable, metrik, atau status deployment.
- Jangan menghapus fungsionalitas yang sudah berjalan.
- Kerjakan perubahan bertahap, jalankan test setelah setiap tahap, dan perbaiki error sebelum lanjut.

Mulai dari audit dan tampilkan rencana singkat, kemudian langsung implementasikan perubahan yang aman secara bertahap dan jalankan pengujian lokal. Jika ada perubahan berisiko atau keputusan yang membutuhkan akses akun, jelaskan blocker dan berikan instruksi manual tanpa meminta atau mencetak kredensial.
```

## 15.4 Prosedur deployment melalui GitHub

Setelah Agent menyatakan persiapan lokal lolos:

1. Pastikan proyek sudah masuk repository GitHub dan seluruh perubahan sudah di-commit.
2. Buka Vercel dan pilih **Add New → Project**.
3. Import repository GitHub yang berisi proyek ini.
4. Pastikan **Root Directory** menunjuk ke folder yang berisi `app.py` dan `requirements.txt`.
5. Periksa deteksi framework Flask/Python. Jangan menambahkan build command atau output directory dari framework lain.
6. Atur Python version dan environment variables hanya sesuai konfigurasi yang benar-benar diperlukan proyek.
7. Jalankan deployment Preview terlebih dahulu.
8. Buka URL Preview dan uji halaman utama, form prediksi, hasil probabilitas, halaman evaluasi, dan aset statis.
9. Periksa Function Logs apabila ada error import, model tidak ditemukan, dependency gagal dimuat, atau route gagal.
10. Jika Preview lolos, deploy branch produksi atau jalankan deployment Production.

Alternatif melalui Vercel CLI dari root proyek:

```bash
npm install -g vercel
vercel login
vercel link
vercel dev
vercel
```

Setelah preview deployment diperiksa dan dinyatakan layak, deploy production:

```bash
vercel --prod
```

Jangan jalankan perintah production sebelum preview diuji. Deployment produksi harus diverifikasi kembali melalui URL yang dikembalikan Vercel.

## 15.5 Checklist khusus Vercel

- [ ] `app.py` berada pada entry point yang dapat dideteksi dan mengekspor `app`.
- [ ] Aplikasi tidak bergantung pada `app.run()` untuk melayani request.
- [ ] `requirements.txt` minimal, terkunci pada versi yang telah diuji, dan instalasinya berhasil.
- [ ] Versi Python didukung oleh Vercel dan kompatibel dengan dependency.
- [ ] Model Joblib disertakan dalam deployment bundle dan dapat dimuat dari path yang benar.
- [ ] Pipeline preprocessing dan Logistic Regression tetap konsisten dengan training.
- [ ] Dataset mentah dan notebook tidak masuk bundle runtime bila tidak diperlukan.
- [ ] Tidak ada penulisan file permanen ke filesystem saat request.
- [ ] Aset statis dapat dimuat dan tidak menghasilkan 404.
- [ ] Tidak ada secret atau file `.env` yang di-commit.
- [ ] GET dan POST routes lulus tes.
- [ ] Satu prediksi end-to-end berhasil pada environment lokal Vercel.
- [ ] Preview deployment berhasil dan diuji sebelum production.
- [ ] Function Logs tidak menunjukkan error saat smoke test.
- [ ] README memuat instruksi deploy serta keterbatasan proyek.

**Definisi selesai:** proyek baru boleh disebut siap deploy ketika pemeriksaan lokal menggunakan konfigurasi Vercel lulus, artefak model dapat dimuat, seluruh route dan aset bekerja, dan Preview deployment yang sebenarnya telah berhasil diuji. Jika belum ada URL deployment, laporkan sebagai "siap diuji deploy" dan bukan "sudah live".
