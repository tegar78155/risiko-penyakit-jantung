# Prediksi Indikasi Penyakit Jantung

Prototipe akademik menggunakan Logistic Regression pada dataset Cleveland.

## Instalasi
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Training
```powershell
python train_model.py
```

## Menjalankan aplikasi
```powershell
python app.py
```
Buka `http://127.0.0.1:5000`.

## Tes
```powershell
python -m pytest -q
```

## Keterbatasan
Dataset hanya 303 sampel. Aplikasi bukan alat diagnosis medis.
" filePath">
D:\Unduhan\Prediksi Risiko Penyakit Jantung Berdasarkan Kondisi Kesehatan\README.md