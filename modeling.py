from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "flag_gagal_bayar"
THRESHOLD = 0.415

NUM = [
    "omzet_bulanan",
    "lama_usaha_tahun",
    "jumlah_karyawan",
    "plafon_kredit",
    "skor_kredit_internal",
    "rasio_cicilan_omzet",
    "jumlah_tunggakan_historis",
    "biaya_operasional_bulanan",
    "pertumbuhan_omzet_yoy",
]

# Hasil seleksi fitur pada notebook: B_tanpa_plafon.
MODEL_NUM = [c for c in NUM if c != "plafon_kredit"]

CAT = [
    "sektor_usaha",
    "jenis_produk_kredit",
    "kepemilikan_agunan",
    "status_kepemilikan_tempat_usaha",
    "channel_pengajuan_kredit",
]

INPUT_FEATURES = NUM + CAT
LGD_SCENARIO = {"ya": 0.40, "tidak": 0.75}


class GroupImputer(BaseEstimator, TransformerMixin):
    """Imputasi mean per sektor seperti konfigurasi final notebook."""

    def __init__(self, method: str = "mean_sector"):
        self.method = method

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.global_ = X[NUM].mean() if self.method == "mean_sector" else X[NUM].median()
        if self.global_.isna().any():
            raise ValueError("Ada kolom numerik yang seluruh nilainya kosong pada training data.")
        grouped = X.groupby("sektor_usaha", dropna=False)[NUM]
        self.group_ = grouped.mean() if self.method == "mean_sector" else grouped.median()
        return self

    def transform(self, X: pd.DataFrame):
        z = X.copy()
        for c in NUM:
            if c not in z.columns:
                z[c] = np.nan
            if self.method != "median_global" and "sektor_usaha" in z.columns:
                fill = z["sektor_usaha"].map(self.group_[c])
            else:
                fill = pd.Series(np.nan, index=z.index)
            z[c] = pd.to_numeric(z[c], errors="coerce").fillna(fill).fillna(self.global_[c])
        for c in CAT:
            if c not in z.columns:
                z[c] = "Tidak diketahui"
            z[c] = z[c].fillna("Tidak diketahui")
        return z


class DerivedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy()


class DistributionTreatment(BaseEstimator, TransformerMixin):
    """Dibiarkan untuk menjaga struktur pipeline notebook; treatment final = none."""

    def __init__(self, method: str = "none", columns=()):
        self.method = method
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy()


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaning inti yang mengikuti notebook sebelum pemodelan."""
    df = raw.drop_duplicates().copy()

    for c in NUM:
        if c not in df.columns:
            df[c] = np.nan
        before = df[c]
        numeric = pd.to_numeric(before, errors="coerce")
        invalid = numeric.notna() & ~np.isfinite(numeric)
        if c == "pertumbuhan_omzet_yoy":
            invalid |= numeric.lt(-100)
        elif c in ["omzet_bulanan", "plafon_kredit"]:
            invalid |= numeric.le(0)
        else:
            invalid |= numeric.lt(0)
        if c in ["jumlah_karyawan", "jumlah_tunggakan_historis"]:
            invalid |= numeric.notna() & ~np.isclose(numeric, numeric.round())
        df[c] = numeric.mask(invalid)

    for c in CAT + ["jenis_kelamin_pemilik_usaha"]:
        if c in df.columns:
            s = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).replace("", pd.NA)
            df[c] = s.astype(object).where(s.notna(), np.nan)

    if "id_debitur" not in df.columns:
        df["id_debitur"] = [f"ROW-{i+1:05d}" for i in range(len(df))]

    return df


def validate_training_data(df: pd.DataFrame) -> None:
    missing = [c for c in INPUT_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Kolom training belum lengkap: " + ", ".join(missing))
    if not df[TARGET].dropna().isin([0, 1]).all():
        raise ValueError(f"{TARGET} harus bernilai 0 atau 1.")


def build_final_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("num", StandardScaler(), MODEL_NUM)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = LogisticRegression(
        C=1,
        solver="liblinear",
        max_iter=3000,
        random_state=SEED,
    )
    return Pipeline(
        [
            ("imputer", GroupImputer("mean_sector")),
            ("features", DerivedFeatures(tuple(MODEL_NUM))),
            ("treatment", DistributionTreatment("none", tuple(MODEL_NUM))),
            ("pre", pre),
            ("model", model),
        ]
    )


@dataclass
class ModelBundle:
    pipeline: Pipeline
    threshold: float
    cleaned: pd.DataFrame
    scored: pd.DataFrame
    test_metrics: Dict[str, float]
    confusion: np.ndarray


def add_credit_risk_outputs(
    df: pd.DataFrame,
    pipeline: Pipeline,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    out = df.copy()
    pd_est = pipeline.predict_proba(out[INPUT_FEATURES])[:, 1]
    out["pd_estimasi_model"] = pd_est
    out["flag_prediksi_review"] = (pd_est >= threshold).astype(int)

    collateral = out["kepemilikan_agunan"].astype("string").str.strip().str.lower()
    out["lgd_skenario"] = collateral.map(LGD_SCENARIO).fillna(0.75)
    out["ead_proxy"] = pd.to_numeric(out["plafon_kredit"], errors="coerce")
    out["ecl_estimasi"] = out["pd_estimasi_model"] * out["lgd_skenario"] * out["ead_proxy"]
    out["risk_band"] = pd.cut(
        out["pd_estimasi_model"],
        bins=[-np.inf, 0.05, 0.15, threshold, np.inf],
        labels=["Rendah", "Menengah", "Tinggi", "Perlu Review"],
        right=False,
    ).astype(str)
    return out


def train_from_dataframe(raw: pd.DataFrame) -> ModelBundle:
    df = clean_data(raw)
    validate_training_data(df)

    # Sama seperti notebook: split 80:20, stratified, random_state=42.
    ids_train, ids_test = train_test_split(
        df.index,
        test_size=0.2,
        stratify=df[TARGET].astype(int),
        random_state=SEED,
    )
    X_train = df.loc[ids_train, INPUT_FEATURES].copy()
    y_train = df.loc[ids_train, TARGET].astype(int)
    X_test = df.loc[ids_test, INPUT_FEATURES].copy()
    y_test = df.loc[ids_test, TARGET].astype(int)

    pipe = build_final_pipeline()
    pipe.fit(X_train, y_train)

    test_pd = pipe.predict_proba(X_test)[:, 1]
    test_flag = (test_pd >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_flag, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_test, test_flag),
        "precision": precision_score(y_test, test_flag, zero_division=0),
        "recall": recall_score(y_test, test_flag, zero_division=0),
        "f1": f1_score(y_test, test_flag, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_pd),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    scored = add_credit_risk_outputs(df, pipe, THRESHOLD)
    return ModelBundle(
        pipeline=pipe,
        threshold=THRESHOLD,
        cleaned=df,
        scored=scored,
        test_metrics=metrics,
        confusion=np.array([[tn, fp], [fn, tp]]),
    )


def score_one(bundle: ModelBundle, record: Dict) -> Dict[str, float]:
    row = pd.DataFrame([record])
    for c in INPUT_FEATURES:
        if c not in row.columns:
            row[c] = np.nan if c in NUM else "Tidak diketahui"
    scored = add_credit_risk_outputs(row, bundle.pipeline, bundle.threshold).iloc[0]
    return {
        "pd": float(scored["pd_estimasi_model"]),
        "threshold": float(bundle.threshold),
        "flag_review": int(scored["flag_prediksi_review"]),
        "lgd": float(scored["lgd_skenario"]),
        "ead": float(scored["ead_proxy"]),
        "ecl": float(scored["ecl_estimasi"]),
        "risk_band": str(scored["risk_band"]),
    }
