from __future__ import annotations

from io import BytesIO
from pathlib import Path
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from modeling import (
    CAT,
    INPUT_FEATURES,
    MODEL_NUM,
    NUM,
    TARGET,
    THRESHOLD,
    ModelBundle,
    score_one,
    train_from_dataframe,
)

st.set_page_config(
    page_title="Proksima Credit Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_URL = os.getenv(
    "PROKSIMA_DATA_URL",
    "https://drive.google.com/uc?export=download&id=1Bm5ot3U-EB9THbUqlFvjfMKry8sc2nEx",
)
LOCAL_DATA = Path("data/portfolio.csv")


def rupiah(x: float) -> str:
    if pd.isna(x):
        return "Rp0"
    x = float(x)
    if abs(x) >= 1e12:
        return f"Rp{x/1e12:,.2f} T"
    if abs(x) >= 1e9:
        return f"Rp{x/1e9:,.2f} M"
    if abs(x) >= 1e6:
        return f"Rp{x/1e6:,.1f} jt"
    return f"Rp{x:,.0f}"


def pct(x: float) -> str:
    return f"{float(x):.2%}"


@st.cache_data(show_spinner=False)
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(content))


@st.cache_data(show_spinner=False, ttl=3600)
def download_default_data(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type.lower():
        raise ValueError("URL data mengembalikan halaman HTML, bukan CSV. Pastikan file Google Drive dapat diakses publik.")
    return pd.read_csv(BytesIO(response.content))


@st.cache_resource(show_spinner="Menyiapkan model Logistic Regression dari konfigurasi notebook...")
def train_cached(csv_bytes: bytes) -> ModelBundle:
    raw = read_csv_bytes(csv_bytes)
    return train_from_dataframe(raw)


def dataframe_to_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------- Sidebar: sumber data ----------
st.sidebar.header("Sumber Data")
uploaded = st.sidebar.file_uploader(
    "Upload CSV portofolio (opsional)",
    type=["csv"],
    help="Jika tidak diupload, aplikasi mencoba data/portfolio.csv lalu URL Google Drive dari notebook.",
)

raw = None
source_label = ""
load_error = None

try:
    if uploaded is not None:
        raw = read_csv_bytes(uploaded.getvalue())
        source_label = f"Upload: {uploaded.name}"
    elif LOCAL_DATA.exists():
        raw = pd.read_csv(LOCAL_DATA)
        source_label = "Repository: data/portfolio.csv"
    else:
        raw = download_default_data(DEFAULT_DATA_URL)
        source_label = "Google Drive dataset notebook"
except Exception as exc:
    load_error = exc

if raw is None:
    st.title("Proksima Credit Risk Dashboard")
    st.error("Dataset belum dapat dimuat.")
    st.info(
        "Upload CSV melalui sidebar, atau tambahkan file `data/portfolio.csv` ke repository. "
        "Jika memakai Google Drive, pastikan file CSV dapat diakses publik."
    )
    if load_error:
        st.code(str(load_error))
    st.stop()

required = set(INPUT_FEATURES + [TARGET])
missing = sorted(required - set(raw.columns))
if missing:
    st.error("CSV belum sesuai struktur notebook. Kolom yang belum ada: " + ", ".join(missing))
    st.stop()

bundle = train_cached(dataframe_to_bytes(raw))
df = bundle.scored.copy()

st.sidebar.success(source_label)
st.sidebar.caption(f"Model final: Logistic Regression · threshold {bundle.threshold:.3f}")

# ---------- Header ----------
st.title("Proksima · Credit Risk & ECL Dashboard")
st.caption(
    "Dashboard interaktif berdasarkan notebook Final Project Proksima. "
    "PD berasal dari Logistic Regression; ECL menggunakan PD × LGD × EAD, dengan EAD = plafon kredit."
)

# ---------- Global filters ----------
st.sidebar.header("Filter Portofolio")
sector_values = sorted(df["sektor_usaha"].dropna().astype(str).unique().tolist())
selected_sector = st.sidebar.multiselect("Sektor usaha", sector_values, default=sector_values)

product_values = sorted(df["jenis_produk_kredit"].dropna().astype(str).unique().tolist())
selected_product = st.sidebar.multiselect("Produk kredit", product_values, default=product_values)

collateral_values = sorted(df["kepemilikan_agunan"].dropna().astype(str).unique().tolist())
selected_collateral = st.sidebar.multiselect("Agunan", collateral_values, default=collateral_values)

max_pd = st.sidebar.slider("Maksimum PD", 0.0, 1.0, 1.0, 0.01)

filtered = df[
    df["sektor_usaha"].astype(str).isin(selected_sector)
    & df["jenis_produk_kredit"].astype(str).isin(selected_product)
    & df["kepemilikan_agunan"].astype(str).isin(selected_collateral)
    & (df["pd_estimasi_model"] <= max_pd)
].copy()

# ---------- Tabs ----------
tab_overview, tab_search, tab_new, tab_model = st.tabs(
    ["Overview", "Cari Nasabah", "Analisis Nasabah Baru", "Model & Metodologi"]
)

with tab_overview:
    if filtered.empty:
        st.warning("Tidak ada data yang sesuai filter.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Debitur", f"{len(filtered):,}")
        c2.metric("Rata-rata PD", pct(filtered["pd_estimasi_model"].mean()))
        c3.metric("Perlu Review", pct(filtered["flag_prediksi_review"].mean()))
        c4.metric("Total EAD", rupiah(filtered["ead_proxy"].sum()))
        c5.metric("Expected Credit Loss", rupiah(filtered["ecl_estimasi"].sum()))

        left, right = st.columns(2)
        with left:
            fig_pd = px.histogram(
                filtered,
                x="pd_estimasi_model",
                nbins=35,
                title="Distribusi Probability of Default",
                labels={"pd_estimasi_model": "Probability of Default", "count": "Jumlah debitur"},
            )
            fig_pd.add_vline(
                x=bundle.threshold,
                line_dash="dash",
                annotation_text=f"Threshold {bundle.threshold:.3f}",
                annotation_position="top right",
            )
            fig_pd.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig_pd, use_container_width=True)

        with right:
            by_sector = (
                filtered.groupby("sektor_usaha", as_index=False)
                .agg(ecl_estimasi=("ecl_estimasi", "sum"), ead_proxy=("ead_proxy", "sum"), pd_mean=("pd_estimasi_model", "mean"))
                .sort_values("ecl_estimasi")
            )
            fig_sector = px.bar(
                by_sector,
                x="ecl_estimasi",
                y="sektor_usaha",
                orientation="h",
                title="Expected Credit Loss per Sektor",
                labels={"ecl_estimasi": "ECL (Rp)", "sektor_usaha": "Sektor"},
                hover_data={"pd_mean": ":.2%", "ead_proxy": ":,.0f"},
            )
            fig_sector.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig_sector, use_container_width=True)

        left, right = st.columns(2)
        with left:
            sector_risk = (
                filtered.groupby("sektor_usaha", as_index=False)
                .agg(
                    actual_default_rate=(TARGET, "mean"),
                    pd_mean=("pd_estimasi_model", "mean"),
                    n=("id_debitur", "size"),
                )
            )
            melted = sector_risk.melt(
                id_vars=["sektor_usaha", "n"],
                value_vars=["actual_default_rate", "pd_mean"],
                var_name="metric",
                value_name="rate",
            )
            melted["metric"] = melted["metric"].map(
                {"actual_default_rate": "Default Aktual", "pd_mean": "Rata-rata PD"}
            )
            fig_compare = px.bar(
                melted,
                x="sektor_usaha",
                y="rate",
                color="metric",
                barmode="group",
                title="Default Aktual vs Rata-rata PD per Sektor",
                labels={"rate": "Rate", "sektor_usaha": "Sektor", "metric": ""},
                hover_data={"n": True},
            )
            fig_compare.update_yaxes(tickformat=".0%")
            fig_compare.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig_compare, use_container_width=True)

        with right:
            sample = filtered
            if len(sample) > 3000:
                sample = sample.sample(3000, random_state=42)
            fig_scatter = px.scatter(
                sample,
                x="pd_estimasi_model",
                y="ecl_estimasi",
                size="ead_proxy",
                color="risk_band",
                hover_name="id_debitur",
                hover_data=["sektor_usaha", "plafon_kredit", "kepemilikan_agunan"],
                title="Hubungan PD, ECL, dan Eksposur Kredit",
                labels={"pd_estimasi_model": "PD", "ecl_estimasi": "ECL (Rp)", "risk_band": "Risk Band"},
                size_max=28,
            )
            fig_scatter.update_xaxes(tickformat=".0%")
            fig_scatter.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Portofolio Terfilter")
        name_col = "nama_nasabah" if "nama_nasabah" in filtered.columns else None
        show_cols = ["id_debitur"]
        if name_col:
            show_cols.append(name_col)
        show_cols += [
            "sektor_usaha",
            "jenis_produk_kredit",
            "plafon_kredit",
            "pd_estimasi_model",
            "flag_prediksi_review",
            "lgd_skenario",
            "ecl_estimasi",
            "risk_band",
        ]
        table = filtered[show_cols].sort_values("pd_estimasi_model", ascending=False).copy()
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "pd_estimasi_model": st.column_config.ProgressColumn("PD", min_value=0.0, max_value=1.0, format="%.2f"),
                "plafon_kredit": st.column_config.NumberColumn("Plafon Kredit", format="Rp %.0f"),
                "lgd_skenario": st.column_config.NumberColumn("LGD", format="%.2f"),
                "ecl_estimasi": st.column_config.NumberColumn("ECL", format="Rp %.0f"),
            },
        )

with tab_search:
    st.subheader("Pencarian Nasabah")
    has_name = "nama_nasabah" in df.columns
    if has_name:
        st.caption("Pencarian dapat menggunakan nama nasabah atau ID debitur.")
        q = st.text_input("Cari nama / ID", placeholder="Contoh: Siti atau UMKM-06253")
        mask = (
            df["nama_nasabah"].astype(str).str.contains(q, case=False, na=False)
            | df["id_debitur"].astype(str).str.contains(q, case=False, na=False)
        ) if q else pd.Series(False, index=df.index)
    else:
        st.info(
            "Dataset notebook tidak mempunyai kolom `nama_nasabah`, sehingga data yang ada hanya bisa dicari dengan `id_debitur`. "
            "Jika CSV berikutnya menambahkan `nama_nasabah`, dashboard otomatis mengaktifkan pencarian nama."
        )
        q = st.text_input("Cari ID debitur", placeholder="Contoh: UMKM-06253")
        mask = df["id_debitur"].astype(str).str.contains(q, case=False, na=False) if q else pd.Series(False, index=df.index)

    results = df.loc[mask].copy()
    if q and results.empty:
        st.warning("Nasabah tidak ditemukan.")
    elif not results.empty:
        selector_cols = ["id_debitur"] + (["nama_nasabah"] if has_name else [])
        options = results[selector_cols].astype(str).agg(" · ".join, axis=1).tolist()
        choice = st.selectbox("Pilih nasabah", options)
        row = results.iloc[options.index(choice)]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Probability of Default", pct(row["pd_estimasi_model"]))
        c2.metric("Status", "PERLU REVIEW" if row["flag_prediksi_review"] == 1 else "Tidak ditandai")
        c3.metric("EAD", rupiah(row["ead_proxy"]))
        c4.metric("Expected Credit Loss", rupiah(row["ecl_estimasi"]))

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(row["pd_estimasi_model"] * 100),
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": "Probability of Default"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "threshold": {
                        "line": {"width": 4},
                        "thickness": 0.8,
                        "value": bundle.threshold * 100,
                    },
                },
            )
        )
        gauge.update_layout(height=300, margin=dict(l=30, r=30, t=55, b=20))
        st.plotly_chart(gauge, use_container_width=True)

        details = {
            "ID Debitur": row.get("id_debitur", "-"),
            "Nama Nasabah": row.get("nama_nasabah", "Tidak tersedia pada dataset notebook"),
            "Sektor": row.get("sektor_usaha", "-"),
            "Produk Kredit": row.get("jenis_produk_kredit", "-"),
            "Agunan": row.get("kepemilikan_agunan", "-"),
            "Omzet Bulanan": rupiah(row.get("omzet_bulanan", 0)),
            "Biaya Operasional": rupiah(row.get("biaya_operasional_bulanan", 0)),
            "Plafon Kredit": rupiah(row.get("plafon_kredit", 0)),
            "Skor Kredit Internal": row.get("skor_kredit_internal", "-"),
            "Rasio Cicilan / Omzet": f"{row.get('rasio_cicilan_omzet', 0):.1f}%",
            "Tunggakan Historis": int(row.get("jumlah_tunggakan_historis", 0)),
            "Pertumbuhan Omzet YoY": f"{row.get('pertumbuhan_omzet_yoy', 0):.1f}%",
            "LGD Skenario": pct(row.get("lgd_skenario", 0)),
            "Risk Band": row.get("risk_band", "-"),
        }
        st.dataframe(pd.DataFrame(details.items(), columns=["Atribut", "Nilai"]), hide_index=True, use_container_width=True)

with tab_new:
    st.subheader("Analisis Nasabah Baru")
    st.caption(
        "Masukkan data finansial dan profil kredit. Model akan menghasilkan PD, status review, LGD skenario, EAD, dan ECL. "
        "Nama nasabah hanya untuk identifikasi dan tidak menjadi predictor model."
    )

    sector_options = sorted(df["sektor_usaha"].dropna().astype(str).unique().tolist())
    product_options = sorted(df["jenis_produk_kredit"].dropna().astype(str).unique().tolist())
    collateral_options = sorted(df["kepemilikan_agunan"].dropna().astype(str).unique().tolist())
    ownership_options = sorted(df["status_kepemilikan_tempat_usaha"].dropna().astype(str).unique().tolist())
    channel_options = sorted(df["channel_pengajuan_kredit"].dropna().astype(str).unique().tolist())

    med = df[NUM].median(numeric_only=True)

    with st.form("new_customer_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        nama = r1c1.text_input("Nama nasabah", placeholder="Contoh: Budi Santoso")
        customer_id = r1c2.text_input("ID debitur", placeholder="Contoh: NEW-001")
        sektor = r1c3.selectbox("Sektor usaha", sector_options)

        r2c1, r2c2, r2c3 = st.columns(3)
        omzet = r2c1.number_input("Omzet bulanan (Rp)", min_value=1.0, value=float(max(med.get("omzet_bulanan", 1), 1)), step=1_000_000.0)
        biaya = r2c2.number_input("Biaya operasional bulanan (Rp)", min_value=0.0, value=float(max(med.get("biaya_operasional_bulanan", 0), 0)), step=1_000_000.0)
        plafon = r2c3.number_input("Plafon kredit / EAD proxy (Rp)", min_value=1.0, value=float(max(med.get("plafon_kredit", 1), 1)), step=5_000_000.0)

        r3c1, r3c2, r3c3 = st.columns(3)
        lama = r3c1.number_input("Lama usaha (tahun)", min_value=0.0, value=float(max(med.get("lama_usaha_tahun", 0), 0)), step=0.5)
        karyawan = r3c2.number_input("Jumlah karyawan", min_value=0, value=int(max(round(med.get("jumlah_karyawan", 0)), 0)), step=1)
        skor = r3c3.number_input("Skor kredit internal", min_value=0.0, value=float(max(med.get("skor_kredit_internal", 0), 0)), step=1.0)

        r4c1, r4c2, r4c3 = st.columns(3)
        rasio = r4c1.number_input("Rasio cicilan / omzet (%)", min_value=0.0, value=float(max(med.get("rasio_cicilan_omzet", 0), 0)), step=0.5)
        tunggakan = r4c2.number_input("Jumlah tunggakan historis", min_value=0, value=int(max(round(med.get("jumlah_tunggakan_historis", 0)), 0)), step=1)
        growth = r4c3.number_input("Pertumbuhan omzet YoY (%)", min_value=-100.0, value=float(med.get("pertumbuhan_omzet_yoy", 0)), step=0.5)

        r5c1, r5c2, r5c3, r5c4 = st.columns(4)
        produk = r5c1.selectbox("Jenis produk kredit", product_options)
        agunan = r5c2.selectbox("Kepemilikan agunan", collateral_options)
        ownership = r5c3.selectbox("Status tempat usaha", ownership_options)
        channel = r5c4.selectbox("Channel pengajuan", channel_options)

        submitted = st.form_submit_button("Hitung Risiko Kredit", type="primary", use_container_width=True)

    if submitted:
        record = {
            "id_debitur": customer_id or "NEW",
            "nama_nasabah": nama,
            "sektor_usaha": sektor,
            "omzet_bulanan": omzet,
            "lama_usaha_tahun": lama,
            "jumlah_karyawan": karyawan,
            "plafon_kredit": plafon,
            "skor_kredit_internal": skor,
            "rasio_cicilan_omzet": rasio,
            "jumlah_tunggakan_historis": tunggakan,
            "jenis_produk_kredit": produk,
            "biaya_operasional_bulanan": biaya,
            "pertumbuhan_omzet_yoy": growth,
            "kepemilikan_agunan": agunan,
            "status_kepemilikan_tempat_usaha": ownership,
            "channel_pengajuan_kredit": channel,
        }
        result = score_one(bundle, record)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("PD", pct(result["pd"]))
        c2.metric("Threshold", pct(result["threshold"]))
        c3.metric("LGD", pct(result["lgd"]))
        c4.metric("EAD", rupiah(result["ead"]))
        c5.metric("ECL", rupiah(result["ecl"]))

        if result["flag_review"]:
            st.warning("Hasil model: **PERLU REVIEW**. Ini adalah flag untuk peninjauan, bukan keputusan otomatis menolak kredit.")
        else:
            st.success("Hasil model: **tidak melewati threshold review**. Keputusan kredit tetap memerlukan kebijakan dan verifikasi bank.")

        st.write(f"Risk band: **{result['risk_band']}**")
        st.latex(r"ECL = PD \times LGD \times EAD")
        st.code(
            f"ECL = {result['pd']:.4f} × {result['lgd']:.2f} × Rp{result['ead']:,.0f} = Rp{result['ecl']:,.0f}",
            language="text",
        )

with tab_model:
    st.subheader("Konfigurasi Final dari Notebook")
    config = pd.DataFrame(
        [
            ["Algoritma", "Logistic Regression"],
            ["Imputasi", "Mean per sektor, fallback mean global training"],
            ["Predictor", ", ".join(MODEL_NUM)],
            ["Kategori sebagai predictor", "Tidak digunakan"],
            ["Treatment distribusi", "None"],
            ["Imbalance treatment", "None"],
            ["Threshold review", f"{bundle.threshold:.3f}"],
            ["EAD", "Proxy = plafon_kredit"],
            ["LGD", "40% jika ada agunan; 75% jika tidak"],
        ],
        columns=["Komponen", "Konfigurasi"],
    )
    st.dataframe(config, hide_index=True, use_container_width=True)

    st.subheader("Evaluasi Holdout 20%")
    m = bundle.test_metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", pct(m["accuracy"]))
    c2.metric("Precision", pct(m["precision"]))
    c3.metric("Recall", pct(m["recall"]))
    c4.metric("F1", pct(m["f1"]))
    c5.metric("ROC AUC", f"{m['roc_auc']:.3f}")

    cm = pd.DataFrame(
        bundle.confusion,
        index=["Aktual Lancar", "Aktual Gagal Bayar"],
        columns=["Prediksi Lancar", "Prediksi Review"],
    )
    fig_cm = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix")
    fig_cm.update_layout(height=350, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

    st.info(
        "Catatan metodologis: `plafon_kredit` tidak dipakai sebagai predictor PD pada konfigurasi final, "
        "tetapi tetap dipakai sebagai proxy EAD dalam perhitungan ECL. PD adalah estimasi model, bukan probabilitas yang dijamin benar untuk setiap individu."
    )
