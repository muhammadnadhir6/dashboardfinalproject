# Proksima Credit Risk & ECL Dashboard

Dashboard Streamlit yang dibangun dari notebook **Final Project Proksima Auditable** untuk memvisualisasikan risiko kredit UMKM secara interaktif.

## Fitur

- Static GitHub Pages landing page: `index.html`.

- KPI portofolio: jumlah debitur, rata-rata PD, review rate, total EAD, total ECL.
- Grafik interaktif Plotly: distribusi PD, ECL per sektor, default aktual vs PD, serta hubungan PD–ECL–EAD.
- Pencarian nasabah. Jika dataset memiliki `nama_nasabah`, pencarian dapat dilakukan dengan nama atau ID. Dataset notebook asli hanya memiliki `id_debitur`.
- Form input data finansial nasabah baru.
- Perhitungan Probability of Default (PD), LGD, EAD, dan Expected Credit Loss (ECL).
- Evaluasi model pada holdout test 20%.

## Logika model yang dipertahankan dari notebook

- Final model: Logistic Regression.
- Train/test split: 80/20, stratified, `random_state=42`.
- Imputasi: `mean_sector`.
- Feature set final: numerik saja.
- Predictor PD final:
  - `omzet_bulanan`
  - `lama_usaha_tahun`
  - `jumlah_karyawan`
  - `skor_kredit_internal`
  - `rasio_cicilan_omzet`
  - `jumlah_tunggakan_historis`
  - `biaya_operasional_bulanan`
  - `pertumbuhan_omzet_yoy`
- `plafon_kredit` tidak dipakai sebagai predictor PD, tetapi dipakai sebagai proxy EAD.
- Threshold review: `0.415` dari F1 OOF optimum pada notebook.
- LGD scenario: 40% jika memiliki agunan, 75% jika tidak.
- ECL: `PD × LGD × EAD`.

## Menjalankan lokal

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data

Urutan sumber data dashboard:

1. CSV yang diupload lewat sidebar.
2. `data/portfolio.csv` jika file tersebut ada di repository.
3. Google Drive URL yang digunakan notebook.

Untuk deployment yang stabil, disarankan menaruh dataset sebagai `data/portfolio.csv`, **hanya jika datanya memang aman untuk dipublikasikan**. Jangan unggah data nasabah yang bersifat rahasia atau memiliki PII ke repository publik.

Anda juga dapat mengganti URL dataset melalui environment variable:

```bash
export PROKSIMA_DATA_URL="https://.../portfolio.csv"
```

## Upload ke GitHub

```bash
git init
git add .
git commit -m "Initial Proksima credit risk dashboard"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```


## GitHub Pages

File `index.html` dapat digunakan sebagai landing page repository. Aktifkan melalui **Settings → Pages → Deploy from a branch → main / root**.

Dashboard model penuh tetap berjalan di Streamlit karena `index.html` bersifat statis. Setelah aplikasi Streamlit berhasil dideploy, isi konstanta `STREAMLIT_URL` di bagian bawah `index.html` agar tombol **Buka Dashboard Streamlit** langsung menuju aplikasi.

## Deploy ke Streamlit Community Cloud

1. Push folder project ini ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih repository dan branch `main`.
4. Main file: `app.py`.
5. Deploy.

## Struktur repository

```text
proksima_credit_risk_dashboard/
├── index.html
├── app.py
├── modeling.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── README.md
└── notebooks/
    └── Final_Project_Proksima_Auditable.ipynb
```

## Catatan penggunaan

Dashboard ini adalah alat analitik dan review. Flag risiko **bukan keputusan otomatis untuk menolak kredit**. Untuk penggunaan operasional, model perlu melalui validasi, monitoring drift, governance, dan penyesuaian terhadap kebijakan risiko lembaga.
