# ============================================================
#   STOCK MOVEMENT PREDICTOR — Full Integrated Interface
#   WITH AUTOMATIC OUTLIER DETECTION & TREATMENT
#   WITH BASIC / ADVANCED MODE TOGGLE
#   Sheikh | Internship Project | VTU CSE (IoT + Cyber)
#   Run with:  streamlit run stock_predictor_full.py
#   Install:   pip install streamlit yfinance pandas numpy
#              scikit-learn plotly scipy
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, roc_curve, classification_report
)

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #080c14; color: #e2e8f0; }

section[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1a2744;
}
section[data-testid="stSidebar"] * { color: #c8d6ef !important; }
section[data-testid="stSidebar"] label {
    color: #5b8fc9 !important; font-size: 0.73rem !important;
    font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.07em;
}

/* ── MODE SWITCHER ─────────────────────────────────────── */
.mode-switcher {
    display: flex; gap: 0; margin: 0 0 18px 0;
    background: #0a0e1a; border: 1px solid #1a2744;
    border-radius: 10px; overflow: hidden; padding: 4px; gap: 4px;
}
.mode-btn {
    flex: 1; padding: 10px 8px; text-align: center; border-radius: 7px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; cursor: pointer; transition: all 0.2s;
    border: none; background: transparent;
}
.mode-btn-active-basic {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: #fff !important; box-shadow: 0 3px 12px rgba(59,130,246,0.35);
}
.mode-btn-active-advanced {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: #fff !important; box-shadow: 0 3px 12px rgba(124,58,237,0.35);
}
.mode-btn-inactive {
    color: #334155 !important; background: transparent;
}
.mode-badge-basic {
    display: inline-block; background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.3); color: #3b82f6;
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 3px 10px; border-radius: 20px;
    margin-bottom: 10px;
}
.mode-badge-advanced {
    display: inline-block; background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.3); color: #a78bfa;
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 3px 10px; border-radius: 20px;
    margin-bottom: 10px;
}

.app-title {
    font-family: 'Space Mono', monospace; font-size: 1.55rem; font-weight: 700;
    color: #e2e8f0; margin: 0 0 2px 0; letter-spacing: -0.01em;
}
.app-sub { color: #334155; font-size: 0.8rem; font-weight: 300; margin: 0 0 20px 0; }
.sec-title {
    font-family: 'Space Mono', monospace; font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase; color: #3b82f6;
    margin: 0 0 14px 0; padding-bottom: 8px; border-bottom: 1px solid #1a2744;
}
.stat-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:18px; }
.stat-card {
    flex:1; min-width:110px; background:#0d1220; border:1px solid #1a2744;
    border-radius:10px; padding:14px 16px;
}
.stat-label {
    font-size:0.62rem; font-weight:700; letter-spacing:0.12em;
    text-transform:uppercase; color:#3b82f6; margin-bottom:4px;
}
.stat-value {
    font-family:'Space Mono',monospace; font-size:1.25rem;
    font-weight:700; color:#e2e8f0; line-height:1;
}
.stat-sub { font-size:0.68rem; color:#334155; margin-top:3px; }
.result-up {
    background: linear-gradient(135deg,#042f2e,#064e3b);
    border:1.5px solid #10b981; border-radius:14px;
    padding:28px 22px; text-align:center; margin-bottom:12px;
}
.result-down {
    background: linear-gradient(135deg,#1a0505,#3b0f0f);
    border:1.5px solid #ef4444; border-radius:14px;
    padding:28px 22px; text-align:center; margin-bottom:12px;
}
.direction-label {
    font-family:'Space Mono',monospace; font-size:2.4rem;
    font-weight:700; margin:0 0 4px; line-height:1;
}
.direction-sub { font-size:0.78rem; color:#94a3b8; margin:0 0 12px; }
.conf-bar-wrap {
    background:#0a0e1a; border-radius:5px; height:6px; overflow:hidden; margin:8px 0 5px;
}
.conf-bar-fill { height:100%; border-radius:5px; }
.conf-pct { font-family:'Space Mono',monospace; font-size:0.72rem; color:#64748b; }
.model-tag {
    display:inline-block; font-size:0.62rem; font-weight:700;
    letter-spacing:0.1em; text-transform:uppercase; padding:3px 9px;
    border-radius:20px; margin-bottom:8px;
}
.consensus-up {
    background:rgba(16,185,129,0.07); border:1px solid rgba(16,185,129,0.28);
    border-radius:12px; padding:18px 22px; text-align:center;
}
.consensus-down {
    background:rgba(239,68,68,0.07); border:1px solid rgba(239,68,68,0.28);
    border-radius:12px; padding:18px 22px; text-align:center;
}
.consensus-label {
    font-family:'Space Mono',monospace; font-size:1.05rem; font-weight:700; margin:0 0 3px;
}
.consensus-sub { font-size:0.75rem; color:#64748b; }
.info-box {
    background:rgba(59,130,246,0.07); border-left:3px solid #3b82f6;
    border-radius:0 8px 8px 0; padding:10px 14px;
    font-size:0.8rem; color:#93c5fd; margin-bottom:18px;
}
.outlier-method-card {
    background:#0d1220; border-radius:10px; padding:16px 18px;
    margin-bottom:10px; border-left:3px solid;
}
.method-badge {
    display:inline-block; font-size:0.6rem; font-weight:700;
    letter-spacing:0.12em; text-transform:uppercase; padding:3px 10px;
    border-radius:20px; margin-bottom:8px;
}

/* ── BASIC MODE PREDICTION CARD ── */
.basic-hero {
    background: linear-gradient(135deg, #0a0e1a, #0d1220);
    border: 1px solid #1a2744; border-radius: 18px;
    padding: 36px 28px; margin-bottom: 20px;
}
.basic-ticker-label {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #3b82f6; margin-bottom: 6px;
}
.basic-ticker {
    font-family: 'Space Mono', monospace; font-size: 2.2rem;
    font-weight: 700; color: #e2e8f0; margin-bottom: 4px;
}
.basic-close {
    font-family: 'Space Mono', monospace; font-size: 1.1rem;
    color: #64748b; margin-bottom: 28px;
}
.basic-divider {
    border: none; border-top: 1px solid #1a2744; margin: 20px 0;
}
.quick-stat-row { display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap; }
.quick-stat {
    flex: 1; min-width: 100px; background: #0a0e1a;
    border: 1px solid #1a2744; border-radius: 10px; padding: 14px 16px; text-align: center;
}
.quick-stat-val {
    font-family: 'Space Mono', monospace; font-size: 1.1rem;
    font-weight: 700; color: #e2e8f0;
}
.quick-stat-lbl {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #334155; margin-top: 4px;
}
.model-accuracy-row {
    display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap;
}
.model-acc-card {
    flex: 1; background: #0a0e1a; border: 1px solid #1a2744;
    border-radius: 10px; padding: 12px 14px; text-align: center;
}

.stButton > button {
    width:100%; background:linear-gradient(135deg,#1d4ed8,#1e40af);
    color:#fff; border:none; border-radius:10px; font-weight:700;
    font-size:0.9rem; letter-spacing:0.06em; padding:0.7rem 1rem;
    transition:all 0.2s; text-transform:uppercase;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#2563eb,#1d4ed8);
    transform:translateY(-1px); box-shadow:0 6px 20px rgba(59,130,246,0.3);
}
.stTabs [data-baseweb="tab-list"] {
    background:#0d1220; border-radius:10px; padding:4px; gap:4px;
    border:1px solid #1a2744; margin-bottom:16px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:7px; color:#4a6fa8 !important;
    font-size:0.78rem; font-weight:600; padding:5px 14px; letter-spacing:0.04em;
}
.stTabs [aria-selected="true"] { background:#1a2744 !important; color:#e2e8f0 !important; }
.perf-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
.perf-card {
    background:#0d1220; border:1px solid #1a2744; border-radius:10px;
    padding:14px; text-align:center;
}
.perf-val { font-family:'Space Mono',monospace; font-size:1.3rem; font-weight:700; color:#e2e8f0; }
.perf-lbl { font-size:0.62rem; color:#334155; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; margin-top:3px; }
.stDataFrame { border:1px solid #1a2744 !important; border-radius:10px; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#080c14; }
::-webkit-scrollbar-thumb { background:#1a2744; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
FEATURES = [
    "Return_1d","Return_5d","Return_10d",
    "SMA_10","SMA_20","SMA_50",
    "EMA_12","EMA_26","MACD","MACD_signal",
    "RSI","BB_width","BB_pos",
    "Price_SMA10_ratio","Price_SMA20_ratio",
    "Volatility_10d","Volume_ratio",
    "HL_range","OC_change","ATR",
]

MODEL_META = {
    "Logistic Regression": {"color": "#3b82f6", "short": "LR"},
    "SVM":                  {"color": "#a78bfa", "short": "SVM"},
    "Random Forest":        {"color": "#34d399", "short": "RF"},
}

PLY = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080c14",
           font=dict(color="#c8d6ef", family="Inter"))
AX  = dict(gridcolor="#1a2744", linecolor="#1a2744", zerolinecolor="#1a2744")


# ─────────────────────────────────────────────────────────────
#  OUTLIER DETECTION & AUTO-TREATMENT ENGINE
# ─────────────────────────────────────────────────────────────

def detect_outliers(series: pd.Series):
    results = {}

    Q1, Q3  = series.quantile(0.25), series.quantile(0.75)
    IQR     = Q3 - Q1
    iqr_mask = (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)
    results["IQR"] = {
        "mask":    iqr_mask,
        "count":   int(iqr_mask.sum()),
        "lower":   Q1 - 1.5 * IQR,
        "upper":   Q3 + 1.5 * IQR,
        "pct":     iqr_mask.mean() * 100,
    }

    z_scores = np.abs(stats.zscore(series.dropna()))
    z_series = pd.Series(z_scores, index=series.dropna().index)
    z_mask   = pd.Series(False, index=series.index)
    z_mask[z_series.index] = z_series > 3
    results["Z-Score"] = {
        "mask":  z_mask,
        "count": int(z_mask.sum()),
        "pct":   z_mask.mean() * 100,
    }

    median    = series.median()
    mad       = np.median(np.abs(series - median))
    mod_z     = 0.6745 * (series - median) / (mad + 1e-9)
    mz_mask   = np.abs(mod_z) > 3.5
    results["Modified Z-Score"] = {
        "mask":  mz_mask,
        "count": int(mz_mask.sum()),
        "pct":   mz_mask.mean() * 100,
    }

    return results


def auto_select_method(series: pd.Series, outlier_results: dict) -> str:
    skewness = float(series.skew())
    kurtosis = float(series.kurtosis())
    n        = len(series.dropna())

    sample   = series.dropna().sample(min(500, n), random_state=42)
    _, p_val = stats.shapiro(sample)
    is_normal = p_val > 0.05

    if abs(skewness) > 1.0:
        method = "IQR"
        reason = (f"High skewness ({skewness:.2f}) detected. "
                  "IQR is skewness-robust and does not assume normality.")
    elif is_normal and abs(kurtosis) < 3:
        method = "Z-Score"
        reason = (f"Data is approximately normal (Shapiro-Wilk p={p_val:.3f}). "
                  "Z-Score performs optimally on normal distributions.")
    else:
        method = "Modified Z-Score"
        reason = (f"Non-normal distribution (p={p_val:.3f}), "
                  f"kurtosis={kurtosis:.2f}. Modified Z-Score (MAD-based) "
                  "is most robust for heavy-tailed / non-normal data.")

    return method, reason, {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "is_normal": is_normal,
        "shapiro_p": p_val,
    }


def treat_outliers(series: pd.Series, mask: pd.Series, treatment: str) -> pd.Series:
    s = series.copy()
    if treatment == "Winsorize":
        lower = s.quantile(0.05)
        upper = s.quantile(0.95)
        s = s.clip(lower=lower, upper=upper)
    elif treatment == "Median Impute":
        med    = s.median()
        s[mask] = med
    elif treatment == "IQR Clip":
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR    = Q3 - Q1
        s      = s.clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
    elif treatment == "Log Transform":
        min_val = s.min()
        shift   = abs(min_val) + 1 if min_val <= 0 else 0
        s       = np.log1p(s + shift)
    return s


def auto_select_treatment(method: str, series: pd.Series) -> str:
    skewness = float(series.skew())

    if method == "IQR":
        if abs(skewness) > 2.0:
            return "Winsorize"
        return "IQR Clip"
    elif method == "Z-Score":
        return "Winsorize"
    else:
        if abs(skewness) > 2.0:
            return "Log Transform"
        return "Median Impute"


def run_outlier_pipeline(df_features: pd.DataFrame):
    treated_df = df_features.copy()
    report     = {}

    for col in df_features.columns:
        series   = df_features[col].dropna()
        if len(series) < 20:
            continue

        outlier_results = detect_outliers(series)
        method, reason, stats_info = auto_select_method(series, outlier_results)
        mask = outlier_results[method]["mask"]

        if int(mask.sum()) == 0:
            report[col] = {
                "outlier_count": 0,
                "method":        method,
                "treatment":     "None",
                "reason":        reason,
                "stats":         stats_info,
                "all_counts":    {m: outlier_results[m]["count"] for m in outlier_results},
            }
            continue

        treatment = auto_select_treatment(method, series)
        treated_df[col] = treat_outliers(df_features[col], mask, treatment)

        report[col] = {
            "outlier_count":    int(mask.sum()),
            "outlier_pct":      float(mask.mean() * 100),
            "method":           method,
            "treatment":        treatment,
            "reason":           reason,
            "stats":            stats_info,
            "all_counts":       {m: outlier_results[m]["count"] for m in outlier_results},
            "before_mean":      float(series.mean()),
            "after_mean":       float(treated_df[col].mean()),
            "before_std":       float(series.std()),
            "after_std":        float(treated_df[col].std()),
            "before_series":    series,
            "after_series":     treated_df[col],
            "mask":             mask,
        }

    return treated_df, report


# ─────────────────────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────────────────────
def _hex_to_rgba(hex_color, alpha=0.13):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


def _generate_synthetic(n=1500, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    rets  = np.random.normal(0.0003, 0.015, n)
    close = 150.0 * np.exp(np.cumsum(rets))
    rng   = close * np.abs(np.random.normal(0.01, 0.005, n))
    return pd.DataFrame({
        "Open":   close - rng * np.random.uniform(-0.5, 0.5, n),
        "High":   close + rng * 0.6,
        "Low":    close - rng * 0.4,
        "Close":  close,
        "Volume": np.random.randint(5_000_000, 50_000_000, n).astype(float),
    }, index=dates)


def _build_features(df):
    d = df.copy()
    for n in [1, 5, 10]:
        d[f"Return_{n}d"] = d["Close"].pct_change(n)
    for w, c in [(10,"SMA_10"),(20,"SMA_20"),(50,"SMA_50")]:
        d[c] = d["Close"].rolling(w).mean()
    d["EMA_12"]      = d["Close"].ewm(span=12, adjust=False).mean()
    d["EMA_26"]      = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]        = d["EMA_12"] - d["EMA_26"]
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    rm = d["Close"].rolling(20).mean()
    rs = d["Close"].rolling(20).std()
    d["BB_upper"] = rm + 2 * rs
    d["BB_lower"] = rm - 2 * rs
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / (rm + 1e-9)
    d["BB_pos"]   = (d["Close"] - d["BB_lower"]) / (d["BB_upper"] - d["BB_lower"] + 1e-9)
    d["Price_SMA10_ratio"] = d["Close"] / (d["SMA_10"] + 1e-9)
    d["Price_SMA20_ratio"] = d["Close"] / (d["SMA_20"] + 1e-9)
    d["Volatility_10d"]    = d["Close"].pct_change().rolling(10).std()
    d["Volume_ratio"]      = d["Volume"] / (d["Volume"].rolling(10).mean() + 1)
    d["HL_range"]          = (d["High"] - d["Low"]) / (d["Close"] + 1e-9)
    d["OC_change"]         = (d["Close"] - d["Open"]) / (d["Open"] + 1e-9)
    hl  = d["High"] - d["Low"]
    hpc = (d["High"] - d["Close"].shift()).abs()
    lpc = (d["Low"]  - d["Close"].shift()).abs()
    d["ATR"]    = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
    d["Target"] = (d["Close"].shift(-1) > d["Close"]).astype(int)
    d.dropna(inplace=True)
    return d


@st.cache_data(show_spinner=False)
def load_and_train(ticker, start, end, use_synthetic):
    if use_synthetic or not HAS_YF:
        raw = _generate_synthetic(1500, 42)
        is_synthetic = True
    else:
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False)
            if raw.empty:
                raw = _generate_synthetic(1500, 42); is_synthetic = True
            else:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw.dropna(inplace=True); is_synthetic = False
        except Exception:
            raw = _generate_synthetic(1500, 42); is_synthetic = True

    data  = _build_features(raw)
    avail = [f for f in FEATURES if f in data.columns]

    feat_df_raw = data[avail].copy()
    feat_df_treated, outlier_report = run_outlier_pipeline(feat_df_raw)

    data_treated = data.copy()
    for col in avail:
        data_treated[col] = feat_df_treated[col]

    X = data_treated[avail].values
    y = data_treated["Target"].values

    split   = int(len(X) * 0.80)
    X_tr    = X[:split]; X_te = X[split:]
    y_tr    = y[:split]; y_te = y[split:]

    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(X_tr)
    Xte_sc  = scaler.transform(X_te)

    trained = {}; perf = {}

    for name, mdl in [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, C=0.1, random_state=42)),
        ("SVM",
         SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
        ("Random Forest",
         RandomForestClassifier(n_estimators=200, max_depth=6,
                                min_samples_leaf=5, class_weight="balanced",
                                random_state=42, n_jobs=-1)),
    ]:
        mdl.fit(Xtr_sc, y_tr)
        yp  = mdl.predict(Xte_sc)
        yb  = mdl.predict_proba(Xte_sc)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, yb)
        trained[name] = mdl
        perf[name] = {
            "accuracy": accuracy_score(y_te, yp),
            "auc":      roc_auc_score(y_te, yb),
            "f1":       f1_score(y_te, yp),
            "preds":    yp, "proba": yb,
            "y_truth":  y_te, "fpr": fpr, "tpr": tpr,
            "cm":       confusion_matrix(y_te, yp),
            "fi":       (mdl.feature_importances_
                         if hasattr(mdl, "feature_importances_") else None),
        }

    last_row   = scaler.transform(data_treated[avail].values[-1:])
    last_close = float(data_treated["Close"].iloc[-1])
    last_date  = data_treated.index[-1].date()

    return dict(
        models=trained, perf=perf,
        data=data_treated, data_raw=data,
        raw=raw, avail=avail,
        last_row=last_row, last_close=last_close, last_date=last_date,
        is_synthetic=is_synthetic, y_te=y_te, Xte_sc=Xte_sc,
        outlier_report=outlier_report,
        feat_df_raw=feat_df_raw, feat_df_treated=feat_df_treated,
    )


# ─────────────────────────────────────────────────────────────
#  PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────
def predict_one(name, model, row):
    prob = float(model.predict_proba(row)[0, 1])
    direction = "UP" if prob >= 0.5 else "DOWN"
    conf = prob if prob >= 0.5 else 1 - prob
    return direction, prob, conf


def render_result_card(name, direction, prob, conf):
    color   = "#10b981" if direction == "UP" else "#ef4444"
    bg_cls  = "result-up" if direction == "UP" else "result-down"
    arrow   = "▲" if direction == "UP" else "▼"
    mc      = MODEL_META.get(name, {}).get("color", "#3b82f6")
    bar_pct = int(conf * 100)
    st.markdown(f"""
    <div class="{bg_cls}">
        <span class="model-tag" style="background:rgba(255,255,255,0.05);
              color:{mc};border:1px solid {mc}44;">{name}</span>
        <p class="direction-label" style="color:{color};">{arrow} {direction}</p>
        <p class="direction-sub">Tomorrow's predicted direction</p>
        <div class="conf-bar-wrap">
            <div class="conf-bar-fill" style="width:{bar_pct}%; background:{color};"></div>
        </div>
        <p class="conf-pct">{bar_pct}% confidence &nbsp;·&nbsp; P(Up) = {prob:.3f}</p>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 14px;">
        <p style="font-family:'Space Mono',monospace;font-size:0.9rem;
                  font-weight:700;color:#3b82f6;margin:0;">⚙️ Configuration</p>
        <p style="font-size:0.68rem;color:#334155;margin:2px 0 0;">Set inputs, then predict</p>
    </div>
    <hr style="border-color:#1a2744;margin:0 0 14px;">
    """, unsafe_allow_html=True)

    # ── MODE SWITCHER ─────────────────────────────────────────
    st.markdown("""
    <p style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;
              text-transform:uppercase;color:#5b8fc9;margin-bottom:8px;">
        Interface Mode
    </p>""", unsafe_allow_html=True)

    app_mode = st.radio(
        label="Interface Mode",
        options=["🔵  Basic", "🟣  Advanced"],
        index=0,
        label_visibility="collapsed",
        horizontal=True,
    )
    is_basic = app_mode == "🔵  Basic"

    if is_basic:
        st.markdown("""
        <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.22);
                    border-radius:8px;padding:10px 12px;margin:8px 0 14px;">
            <div style="font-size:0.62rem;color:#3b82f6;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">
                Basic Mode
            </div>
            <div style="font-size:0.72rem;color:#64748b;line-height:1.5;">
                Shows only the prediction result — clean and focused.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.22);
                    border-radius:8px;padding:10px 12px;margin:8px 0 14px;">
            <div style="font-size:0.62rem;color:#a78bfa;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">
                Advanced Mode
            </div>
            <div style="font-size:0.72rem;color:#64748b;line-height:1.5;">
                Full analysis: EDA, Outlier Treatment, Model Performance &amp; Prediction.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1a2744;margin:0 0 12px;'>", unsafe_allow_html=True)

    ticker = st.text_input("Stock Ticker", value="AAPL",
                            placeholder="e.g. TSLA, MSFT").upper().strip()
    use_synthetic = st.toggle(
        "Use demo data" + ("" if HAS_YF else " (yfinance not available)"),
        value=not HAS_YF,
    )
    if not use_synthetic and HAS_YF:
        col_a, col_b = st.columns(2)
        with col_a: start_date = str(st.date_input("From", value=pd.to_datetime("2018-01-01")))
        with col_b: end_date   = str(st.date_input("To",   value=pd.to_datetime("2024-01-01")))
    else:
        start_date, end_date = "2018-01-01", "2024-01-01"

    st.markdown("<hr style='border-color:#1a2744;'>", unsafe_allow_html=True)
    model_choice = st.selectbox("Prediction Model",
        ["Ensemble (All 3)", "Logistic Regression", "SVM", "Random Forest"])
    st.markdown("<hr style='border-color:#1a2744;'>", unsafe_allow_html=True)
    predict_btn = st.button("📈  PREDICT", use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────────────────────
mode_badge = (
    '<span class="mode-badge-basic">Basic Mode</span>'
    if is_basic else
    '<span class="mode-badge-advanced">Advanced Mode</span>'
)

st.markdown(f"""
<div class="app-title">📈 Stock Movement Predictor</div>
<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
    {mode_badge}
    <p class="app-sub" style="margin:0;">
        Next-day price direction · LR · SVM · Random Forest &nbsp;|&nbsp;
        {ticker} &nbsp;{'· Demo data' if (use_synthetic or not HAS_YF) else '· Live data'}
    </p>
</div>
""", unsafe_allow_html=True)

if not predict_btn and "payload" not in st.session_state:
    st.markdown("""
    <div class="info-box">
        Configure a ticker and model in the sidebar, then click
        <strong>PREDICT</strong> to train models and generate results.
    </div>""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1,"Ticker",ticker), (c2,"Model",model_choice),
        (c3,"Data","Demo" if (use_synthetic or not HAS_YF) else "Live"),
        (c4,"Mode","Basic" if is_basic else "Advanced"),
    ]:
        col.markdown(f"""
        <div style="background:#0d1220;border:1px solid #1a2744;border-radius:10px;
                    padding:18px;text-align:center;">
            <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;
                        text-transform:uppercase;color:#3b82f6;margin-bottom:6px;">{label}</div>
            <div style="font-family:'Space Mono',monospace;font-size:1.05rem;
                        font-weight:700;color:#e2e8f0;">{val}</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
#  LOAD & TRAIN
# ─────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Fetching data, treating outliers & training models…"):
        payload = load_and_train(ticker, start_date, end_date, use_synthetic)
    st.session_state.update({
        "payload": payload, "model_choice": model_choice,
        "ticker": ticker, "use_syn": use_synthetic or not HAS_YF,
        "app_mode": app_mode,
    })

# Keep mode in sync even without re-predict
if "app_mode" in st.session_state and not predict_btn:
    st.session_state["app_mode"] = app_mode

payload      = st.session_state["payload"]
model_choice = st.session_state.get("model_choice", model_choice)
ticker_disp  = st.session_state.get("ticker", ticker)
is_syn       = st.session_state.get("use_syn", True)

models          = payload["models"]
perf            = payload["perf"]
data            = payload["data"]
data_raw        = payload["data_raw"]
avail           = payload["avail"]
last_row        = payload["last_row"]
last_close      = payload["last_close"]
last_date       = payload["last_date"]
outlier_report  = payload["outlier_report"]
feat_df_raw     = payload["feat_df_raw"]
feat_df_treated = payload["feat_df_treated"]

up_days = int(data["Target"].sum())
up_pct  = up_days / len(data) * 100
n_days  = len(data)
ret_1d  = float(data["Return_1d"].iloc[-1]) * 100
ret_pct_total = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100
total_outliers = sum(r.get("outlier_count", 0) for r in outlier_report.values())
model_names    = list(perf.keys())
best_model     = max(model_names, key=lambda n: perf[n]["auc"])

# Pre-compute all predictions
all_preds = {}
for name in model_names:
    d, prob, conf = predict_one(name, models[name], last_row)
    all_preds[name] = {"direction": d, "prob": prob, "conf": conf}

up_votes  = sum(1 for n in model_names if all_preds[n]["direction"] == "UP")
avg_prob  = float(np.mean([all_preds[n]["prob"] for n in model_names]))
consensus = "UP" if up_votes >= 2 else "DOWN"


# ═════════════════════════════════════════════════════════════
#  ██████╗  █████╗ ███████╗██╗ ██████╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗
#  ██╔══██╗██╔══██╗██╔════╝██║██╔════╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝
#  ██████╔╝███████║███████╗██║██║         ██╔████╔██║██║   ██║██║  ██║█████╗
#  ██╔══██╗██╔══██║╚════██║██║██║         ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝
#  ██████╔╝██║  ██║███████║██║╚██████╗    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗
# ═════════════════════════════════════════════════════════════
if is_basic:
    cons_color = "#10b981" if consensus == "UP" else "#ef4444"
    cons_arrow = "▲" if consensus == "UP" else "▼"
    cons_bg    = "result-up" if consensus == "UP" else "result-down"

    # ── Hero prediction card ──────────────────────────────────
    st.markdown(f"""
    <div class="basic-hero">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;
                    flex-wrap:wrap;gap:20px;">
            <!-- Left: ticker info -->
            <div>
                <div class="basic-ticker-label">Analysing</div>
                <div class="basic-ticker">{ticker_disp}</div>
                <div class="basic-close">
                    Last close: <span style="color:#e2e8f0;">${last_close:,.2f}</span>
                    &nbsp;·&nbsp; {last_date}
                    &nbsp;·&nbsp;
                    <span style="color:{'#10b981' if ret_1d>=0 else '#ef4444'};">
                        {"▲" if ret_1d>=0 else "▼"} {ret_1d:+.2f}% today
                    </span>
                </div>
                <div style="font-size:0.7rem;color:#334155;">
                    {"Demo data" if is_syn else f"{start_date} → {end_date}"}
                    &nbsp;·&nbsp; {n_days:,} trading days
                    &nbsp;·&nbsp; Best model: {best_model} (AUC {perf[best_model]['auc']:.3f})
                </div>
            </div>
            <!-- Right: consensus result -->
            <div style="background:{'rgba(16,185,129,0.07)' if consensus=='UP' else 'rgba(239,68,68,0.07)'};
                        border:1.5px solid {cons_color};border-radius:14px;
                        padding:22px 32px;text-align:center;min-width:200px;">
                <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.15em;
                            text-transform:uppercase;color:#64748b;margin-bottom:8px;">
                    Tomorrow's Direction
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:3.2rem;
                            font-weight:700;color:{cons_color};line-height:1;">
                    {cons_arrow} {consensus}
                </div>
                <div style="font-size:0.75rem;color:#64748b;margin-top:8px;">
                    {up_votes}/3 models agree &nbsp;·&nbsp; avg P(Up) = {avg_prob:.3f}
                </div>
                <div style="background:#0a0e1a;border-radius:5px;height:6px;
                            overflow:hidden;margin:12px 0 5px;">
                    <div style="width:{int(avg_prob*100)}%;height:100%;
                                border-radius:5px;background:{cons_color};"></div>
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#64748b;">
                    {int(avg_prob*100)}% average confidence
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Per-model cards (compact) ─────────────────────────────
    st.markdown('<p class="sec-title">Individual Model Predictions</p>', unsafe_allow_html=True)
    cols_pred = st.columns(3)
    for col, name in zip(cols_pred, model_names):
        with col:
            ap    = all_preds[name]
            color = "#10b981" if ap["direction"] == "UP" else "#ef4444"
            arrow = "▲" if ap["direction"] == "UP" else "▼"
            mc    = MODEL_META[name]["color"]
            bar   = int(ap["conf"] * 100)
            col.markdown(f"""
            <div style="background:#0d1220;border:1px solid #1a2744;border-radius:12px;
                        padding:20px 18px;text-align:center;">
                <span style="font-size:0.6rem;font-weight:700;letter-spacing:0.1em;
                             text-transform:uppercase;color:{mc};background:{mc}18;
                             border:1px solid {mc}44;padding:3px 9px;border-radius:20px;">
                    {name}
                </span>
                <div style="font-family:'Space Mono',monospace;font-size:2rem;
                            font-weight:700;color:{color};margin:14px 0 4px;line-height:1;">
                    {arrow} {ap['direction']}
                </div>
                <div style="font-size:0.7rem;color:#64748b;margin-bottom:10px;">
                    P(Up) = {ap['prob']:.3f}
                </div>
                <div style="background:#0a0e1a;border-radius:4px;height:5px;overflow:hidden;">
                    <div style="width:{bar}%;height:100%;background:{color};border-radius:4px;"></div>
                </div>
                <div style="font-size:0.65rem;color:#334155;margin-top:5px;">{bar}% confidence</div>
                <div style="margin-top:10px;padding-top:10px;border-top:1px solid #1a2744;">
                    <span style="font-size:0.6rem;color:#64748b;">AUC </span>
                    <span style="font-family:'Space Mono',monospace;font-size:0.75rem;
                                 color:{mc};">{perf[name]['auc']:.3f}</span>
                    <span style="font-size:0.6rem;color:#334155;margin:0 6px;">·</span>
                    <span style="font-size:0.6rem;color:#64748b;">Acc </span>
                    <span style="font-family:'Space Mono',monospace;font-size:0.75rem;
                                 color:{mc};">{perf[name]['accuracy']:.3f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Quick stats row ───────────────────────────────────────
    st.markdown("""
    <div class="quick-stat-row">
    """, unsafe_allow_html=True)

    rsi_val   = float(data["RSI"].iloc[-1])
    macd_val  = float(data["MACD"].iloc[-1])
    vol_ratio = float(data["Volume_ratio"].iloc[-1])
    bb_pos    = float(data["BB_pos"].iloc[-1])

    rsi_color = "#ef4444" if rsi_val > 70 else "#10b981" if rsi_val < 30 else "#e2e8f0"
    rsi_label = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"

    q1, q2, q3, q4, q5 = st.columns(5)
    for col, lbl, val, sub, color in [
        (q1, "RSI (14d)",    f"{rsi_val:.1f}",  rsi_label,         rsi_color),
        (q2, "MACD",         f"{macd_val:+.3f}", "vs signal",       "#10b981" if macd_val>0 else "#ef4444"),
        (q3, "Volume Ratio", f"{vol_ratio:.2f}x","vs 10d avg",      "#f59e0b" if vol_ratio>1.5 else "#e2e8f0"),
        (q4, "BB Position",  f"{bb_pos:.2f}",   "0=low · 1=high",  "#a78bfa"),
        (q5, "Up Day %",     f"{up_pct:.1f}%",  f"{up_days}/{n_days} days", "#e2e8f0"),
    ]:
        col.markdown(f"""
        <div class="quick-stat">
            <div class="quick-stat-val" style="color:{color};">{val}</div>
            <div class="quick-stat-lbl">{lbl}</div>
            <div style="font-size:0.6rem;color:#334155;margin-top:2px;">{sub}</div>
        </div>""", unsafe_allow_html=True)

    # ── Mini price chart (last 60 days) ───────────────────────
    st.markdown('<p class="sec-title" style="margin-top:20px;">Recent Price (Last 60 Days)</p>',
                unsafe_allow_html=True)
    recent = data.tail(60)
    fig_mini = go.Figure()
    area_color = "#10b981" if float(recent["Close"].iloc[-1]) >= float(recent["Close"].iloc[0]) else "#ef4444"
    fig_mini.add_trace(go.Scatter(
        x=recent.index, y=recent["Close"],
        line=dict(color=area_color, width=2),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(area_color, alpha=0.08),
        name="Close",
    ))
    fig_mini.add_trace(go.Scatter(
        x=recent.index, y=recent["SMA_20"],
        line=dict(color="#3b82f6", width=1.2, dash="dot"),
        name="SMA 20",
    ))
    fig_mini.update_layout(
        **PLY, xaxis=dict(rangeslider=dict(visible=False), **AX),
        yaxis=dict(**AX), height=260,
        margin=dict(t=10,b=10,l=10,r=10), showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    st.plotly_chart(fig_mini, use_container_width=True)

    # ── Disclaimer ────────────────────────────────────────────
    # st.markdown("""
    # <div style="background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);
    #             border-radius:10px;padding:14px 18px;margin-top:8px;">
    #     <span style="font-size:0.62rem;font-weight:700;color:#f59e0b;
    #                  text-transform:uppercase;letter-spacing:0.1em;">⚠ Disclaimer</span>
    #     <p style="font-size:0.75rem;color:#64748b;margin:6px 0 0;line-height:1.6;">
    #         This tool is for <strong style="color:#94a3b8;">educational purposes only</strong>.
    #         Predictions are based on historical patterns and do not constitute financial advice.
    #         Always do your own research before making investment decisions.
    #         Switch to <strong style="color:#a78bfa;">Advanced Mode</strong>
    #         for full model details, outlier analysis, and EDA.
    #     </p>
    # </div>
    # """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  ADVANCED MODE — full 4-tab layout
# ═════════════════════════════════════════════════════════════
else:
    # ── Top stat row ─────────────────────────────────────────
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card">
        <div class="stat-label">Ticker</div>
        <div class="stat-value">{ticker_disp}</div>
        <div class="stat-sub">{"Demo data" if is_syn else f"{start_date} → {end_date}"}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Last Close</div>
        <div class="stat-value">${last_close:,.2f}</div>
        <div class="stat-sub" style="color:{'#10b981' if ret_1d>=0 else '#ef4444'}">
            {"▲" if ret_1d>=0 else "▼"} {ret_1d:+.2f}% today
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Trading Days</div>
        <div class="stat-value">{n_days:,}</div>
        <div class="stat-sub">after feature engineering</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Up Days</div>
        <div class="stat-value">{up_pct:.1f}%</div>
        <div class="stat-sub">{up_days} of {n_days} days UP</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Outliers Treated</div>
        <div class="stat-value" style="color:#f59e0b;">{total_outliers:,}</div>
        <div class="stat-sub">across all features</div>
      </div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Dataset & EDA",
        "🔍  Outlier Treatment",
        "🤖  Model Performance",
        "🎯  Prediction",
    ])

    # ══════════════════════════════════════════════════════════
    #  TAB 1 — DATASET & EDA
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<p class="sec-title">Price History with Moving Averages</p>',
                    unsafe_allow_html=True)
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"],
            increasing_line_color="#10b981", decreasing_line_color="#ef4444",
            name="OHLC", showlegend=False,
        ))
        for col, color, dash in [("SMA_20","#3b82f6","solid"),("SMA_50","#facc15","dot")]:
            fig_price.add_trace(go.Scatter(
                x=data.index, y=data[col],
                line=dict(color=color, width=1.4, dash=dash), name=col,
            ))
        fig_price.update_layout(
            **PLY, xaxis=dict(rangeslider=dict(visible=False), **AX),
            yaxis=dict(**AX), height=340, margin=dict(t=10,b=10,l=10,r=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        )
        st.plotly_chart(fig_price, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="sec-title">Volume</p>', unsafe_allow_html=True)
            vol_colors = np.where(data["Return_1d"] >= 0, "#10b981", "#ef4444").tolist()
            fig_vol = go.Figure(go.Bar(
                x=data.index, y=data["Volume"], marker_color=vol_colors, name="Volume"
            ))
            fig_vol.update_layout(**PLY, xaxis=dict(**AX), yaxis=dict(**AX),
                                   height=210, margin=dict(t=10,b=10,l=10,r=10), showlegend=False)
            st.plotly_chart(fig_vol, use_container_width=True)

        with c2:
            st.markdown('<p class="sec-title">RSI (14-day)</p>', unsafe_allow_html=True)
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=data.index, y=data["RSI"], line=dict(color="#a78bfa", width=1.4), name="RSI"
            ))
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="#ef4444",
                              annotation_text="Overbought", annotation_font_color="#ef4444",
                              annotation_position="right")
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="#10b981",
                              annotation_text="Oversold", annotation_font_color="#10b981",
                              annotation_position="right")
            fig_rsi.update_layout(**PLY, xaxis=dict(**AX), yaxis=dict(range=[0,100],**AX),
                                   height=210, margin=dict(t=10,b=10,l=10,r=10), showlegend=False)
            st.plotly_chart(fig_rsi, use_container_width=True)

        st.markdown('<p class="sec-title">MACD</p>', unsafe_allow_html=True)
        hist_colors = np.where(data["MACD"] - data["MACD_signal"] >= 0,
                               "#10b981", "#ef4444").tolist()
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Bar(x=data.index, y=data["MACD"]-data["MACD_signal"],
                                   marker_color=hist_colors, name="Histogram", opacity=0.6))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD"],
                                       line=dict(color="#3b82f6",width=1.3), name="MACD"))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD_signal"],
                                       line=dict(color="#ef4444",width=1.3,dash="dot"), name="Signal"))
        fig_macd.update_layout(**PLY, xaxis=dict(**AX), yaxis=dict(**AX), height=200,
                                margin=dict(t=10,b=10,l=10,r=10),
                                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)), barmode="overlay")
        st.plotly_chart(fig_macd, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="sec-title">Feature Distribution (UP vs DOWN)</p>',
                        unsafe_allow_html=True)
            feat_sel = st.selectbox("Select feature", avail, index=0, key="feat_sel")
            up_v = data[data["Target"]==1][feat_sel]
            dn_v = data[data["Target"]==0][feat_sel]
            fig_fd = go.Figure()
            fig_fd.add_trace(go.Histogram(x=up_v, name="UP",
                                          marker_color="#10b981", opacity=0.65, nbinsx=45))
            fig_fd.add_trace(go.Histogram(x=dn_v, name="DOWN",
                                          marker_color="#ef4444", opacity=0.65, nbinsx=45))
            fig_fd.update_layout(**PLY, xaxis=dict(**AX), yaxis=dict(**AX), barmode="overlay",
                                  height=260, margin=dict(t=10,b=10,l=10,r=10),
                                  legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)))
            st.plotly_chart(fig_fd, use_container_width=True)

        with c2:
            st.markdown('<p class="sec-title">Class Balance</p>', unsafe_allow_html=True)
            counts = data["Target"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=["UP ▲","DOWN ▼"],
                values=[int(counts.get(1,0)), int(counts.get(0,0))],
                hole=0.58, marker_colors=["#10b981","#ef4444"], textfont_size=11,
            ))
            fig_pie.update_layout(**PLY, height=260, margin=dict(t=10,b=10,l=10,r=10),
                                   legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)),
                                   annotations=[dict(text="Target",font_size=13,
                                                     font_color="#c8d6ef",showarrow=False)])
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<p class="sec-title">Feature Correlation Heatmap</p>', unsafe_allow_html=True)
        corr = data[avail].corr()
        fig_heat = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig_heat.update_layout(**PLY, height=340, margin=dict(t=10,b=10,l=10,r=10),
                                coloraxis_colorbar=dict(tickfont=dict(size=9)))
        fig_heat.update_xaxes(tickfont=dict(size=8), gridcolor="#1a2744")
        fig_heat.update_yaxes(tickfont=dict(size=8), gridcolor="#1a2744")
        st.plotly_chart(fig_heat, use_container_width=True)

        c1, c2 = st.columns([3,2])
        with c1:
            st.markdown('<p class="sec-title">Raw Data Preview (last 10 rows)</p>',
                        unsafe_allow_html=True)
            fmt = {c: "{:.4f}" for c in ["Open","High","Low","Close"] if c in data.columns}
            if "Volume" in data.columns: fmt["Volume"] = "{:,.0f}"
            st.dataframe(data[["Open","High","Low","Close","Volume"]].tail(10)
                         .style.format(fmt), use_container_width=True, height=260)
        with c2:
            st.markdown('<p class="sec-title">Descriptive Stats</p>', unsafe_allow_html=True)
            st.dataframe(data[["Close","RSI","MACD","Volume_ratio"]].describe()
                         .round(3).style.format("{:.3f}"), use_container_width=True, height=260)


    # ══════════════════════════════════════════════════════════
    #  TAB 2 — OUTLIER TREATMENT
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
        <div style="background:#0d1220;border:1px solid #1a2744;border-radius:12px;
                    padding:18px 22px;margin-bottom:20px;">
            <p style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#3b82f6;
                      font-weight:700;letter-spacing:0.15em;text-transform:uppercase;margin:0 0 10px;">
                How Automatic Outlier Treatment Works
            </p>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                <div style="background:#111827;border-radius:8px;padding:14px;">
                    <div style="font-size:0.68rem;font-weight:700;color:#f59e0b;
                                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">
                        Step 1 — Detect
                    </div>
                    <div style="font-size:0.78rem;color:#94a3b8;line-height:1.5;">
                        All 3 methods run on every feature:
                        IQR, Z-Score, and Modified Z-Score (MAD).
                    </div>
                </div>
                <div style="background:#111827;border-radius:8px;padding:14px;">
                    <div style="font-size:0.68rem;font-weight:700;color:#3b82f6;
                                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">
                        Step 2 — Auto-Select
                    </div>
                    <div style="font-size:0.78rem;color:#94a3b8;line-height:1.5;">
                        Skewness test + Shapiro-Wilk normality test
                        determines the best detection method per feature.
                    </div>
                </div>
                <div style="background:#111827;border-radius:8px;padding:14px;">
                    <div style="font-size:0.68rem;font-weight:700;color:#10b981;
                                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">
                        Step 3 — Treat
                    </div>
                    <div style="font-size:0.78rem;color:#94a3b8;line-height:1.5;">
                        Treatment is matched to detection method:
                        Winsorize, IQR Clip, Median Impute, or Log Transform.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="sec-title">Method Selection Logic</p>', unsafe_allow_html=True)
        rules = [
            ("#f59e0b", "IQR + Winsorize / IQR Clip",
             "Skewness > 1.0",
             "Skewed distributions violate Z-Score's normality assumption. IQR uses quartiles which are robust to asymmetry."),
            ("#3b82f6", "Z-Score + Winsorize",
             "Normal distribution (Shapiro-Wilk p > 0.05) & |kurtosis| < 3",
             "When data is normally distributed, Z-Score maximally leverages the known Gaussian properties of the data."),
            ("#a78bfa", "Modified Z-Score + Median Impute / Log Transform",
             "Non-normal, heavy-tailed distribution",
             "MAD (Median Absolute Deviation) is resistant to the influence of outliers themselves, unlike standard deviation."),
        ]
        for color, method, condition, why in rules:
            st.markdown(f"""
            <div class="outlier-method-card" style="border-color:{color};">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                    <span class="method-badge"
                          style="background:{color}22;color:{color};border:1px solid {color}44;">
                        Auto-Selected When
                    </span>
                    <span style="font-size:0.78rem;color:{color};font-weight:600;">{condition}</span>
                </div>
                <div style="font-size:0.85rem;font-weight:700;color:#e2e8f0;margin-bottom:4px;">
                    → {method}
                </div>
                <div style="font-size:0.78rem;color:#64748b;line-height:1.5;">{why}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<p class="sec-title">Outlier Treatment Summary</p>', unsafe_allow_html=True)
        treated_feats   = [k for k, v in outlier_report.items() if v.get("outlier_count", 0) > 0]
        clean_feats     = [k for k, v in outlier_report.items() if v.get("outlier_count", 0) == 0]
        methods_used    = list({v["method"] for k, v in outlier_report.items()
                                if v.get("outlier_count", 0) > 0})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Features Treated",  len(treated_feats))
        m2.metric("Clean Features",    len(clean_feats))
        m3.metric("Total Outliers",    total_outliers)
        m4.metric("Methods Used",      len(methods_used))

        st.markdown('<p class="sec-title">Per-Feature Detection & Treatment Report</p>',
                    unsafe_allow_html=True)
        report_rows = []
        for feat, info in outlier_report.items():
            if info.get("outlier_count", 0) == 0:
                report_rows.append({
                    "Feature":      feat,
                    "IQR Outliers": info["all_counts"].get("IQR", 0),
                    "Z Outliers":   info["all_counts"].get("Z-Score", 0),
                    "MZ Outliers":  info["all_counts"].get("Modified Z-Score", 0),
                    "Auto Method":  info["method"],
                    "Treatment":    info["treatment"],
                    "Skewness":     round(info["stats"]["skewness"], 3),
                    "Normal?":      "✅ Yes" if info["stats"]["is_normal"] else "❌ No",
                    "Status":       "✅ Clean",
                })
            else:
                report_rows.append({
                    "Feature":      feat,
                    "IQR Outliers": info["all_counts"].get("IQR", 0),
                    "Z Outliers":   info["all_counts"].get("Z-Score", 0),
                    "MZ Outliers":  info["all_counts"].get("Modified Z-Score", 0),
                    "Auto Method":  info["method"],
                    "Treatment":    info["treatment"],
                    "Skewness":     round(info["stats"]["skewness"], 3),
                    "Normal?":      "✅ Yes" if info["stats"]["is_normal"] else "❌ No",
                    "Status":       f"⚠️ {info['outlier_count']} treated ({info['outlier_pct']:.1f}%)",
                })
        report_df = pd.DataFrame(report_rows).set_index("Feature")
        st.dataframe(report_df, use_container_width=True, height=380)

        st.markdown('<p class="sec-title">Outlier Count Comparison — All 3 Detection Methods</p>',
                    unsafe_allow_html=True)
        feats_with = [f for f in outlier_report if outlier_report[f].get("outlier_count",0) > 0]
        if feats_with:
            iqr_c  = [outlier_report[f]["all_counts"]["IQR"]              for f in feats_with]
            z_c    = [outlier_report[f]["all_counts"]["Z-Score"]           for f in feats_with]
            mz_c   = [outlier_report[f]["all_counts"]["Modified Z-Score"]  for f in feats_with]

            fig_comp = go.Figure()
            for label, vals, color in [
                ("IQR",             iqr_c, "#f59e0b"),
                ("Z-Score",         z_c,   "#3b82f6"),
                ("Modified Z-Score",mz_c,  "#a78bfa"),
            ]:
                fig_comp.add_trace(go.Bar(
                    name=label, x=feats_with, y=vals, marker_color=color, opacity=0.82,
                ))
            fig_comp.update_layout(
                **PLY, barmode="group",
                xaxis=dict(tickangle=-35, **AX), yaxis=dict(title="Outlier Count", **AX),
                height=320, margin=dict(t=10,b=80,l=10,r=10),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.success("✅ No outliers detected in any feature.")

        st.markdown('<p class="sec-title">Before vs After Treatment — Distribution Comparison</p>',
                    unsafe_allow_html=True)
        if feats_with:
            sel_feat = st.selectbox("Select feature to inspect", feats_with, key="outlier_feat_sel")
            info = outlier_report[sel_feat]

            method_colors = {"IQR":"#f59e0b","Z-Score":"#3b82f6","Modified Z-Score":"#a78bfa"}
            mc = method_colors.get(info["method"], "#3b82f6")
            st.markdown(f"""
            <div class="outlier-method-card" style="border-color:{mc};margin-bottom:16px;">
                <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;">
                    <div>
                        <span style="font-size:0.62rem;color:{mc};font-weight:700;
                                     text-transform:uppercase;letter-spacing:0.1em;">
                            Auto-Selected Method
                        </span>
                        <div style="font-family:'Space Mono',monospace;font-size:1rem;
                                    font-weight:700;color:{mc};margin-top:2px;">
                            {info['method']}
                        </div>
                    </div>
                    <div>
                        <span style="font-size:0.62rem;color:#10b981;font-weight:700;
                                     text-transform:uppercase;letter-spacing:0.1em;">
                            Treatment Applied
                        </span>
                        <div style="font-family:'Space Mono',monospace;font-size:1rem;
                                    font-weight:700;color:#10b981;margin-top:2px;">
                            {info['treatment']}
                        </div>
                    </div>
                    <div style="flex:1;min-width:260px;">
                        <span style="font-size:0.62rem;color:#64748b;font-weight:700;
                                     text-transform:uppercase;letter-spacing:0.1em;">
                            Why This Method Was Chosen
                        </span>
                        <div style="font-size:0.78rem;color:#94a3b8;margin-top:4px;line-height:1.5;">
                            {info['reason']}
                        </div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            before_s = info["before_series"]
            after_s  = info["after_series"]
            mask     = info["mask"]

            c1, c2 = st.columns(2)
            with c1:
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=before_s, name="Before", marker_color="#ef4444", opacity=0.65, nbinsx=50))
                fig_dist.add_trace(go.Histogram(x=after_s,  name="After",  marker_color="#10b981", opacity=0.65, nbinsx=50))
                fig_dist.update_layout(**PLY, barmode="overlay", title=f"{sel_feat} — Distribution",
                                       xaxis=dict(**AX), yaxis=dict(**AX), height=300,
                                       margin=dict(t=40,b=10,l=10,r=10),
                                       legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)))
                st.plotly_chart(fig_dist, use_container_width=True)

            with c2:
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(y=before_s, name="Before", marker_color="#ef4444",
                                         boxmean=True, jitter=0.3, pointpos=-1.8,
                                         marker=dict(size=2, opacity=0.4)))
                fig_box.add_trace(go.Box(y=after_s, name="After", marker_color="#10b981",
                                         boxmean=True, jitter=0.3, pointpos=-1.8,
                                         marker=dict(size=2, opacity=0.4)))
                fig_box.update_layout(**PLY, title=f"{sel_feat} — Box Plot",
                                      xaxis=dict(**AX), yaxis=dict(**AX), height=300,
                                      margin=dict(t=40,b=10,l=10,r=10),
                                      legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)))
                st.plotly_chart(fig_box, use_container_width=True)

            st.markdown('<p class="sec-title" style="margin-top:8px;">Outlier Locations in Time Series</p>',
                        unsafe_allow_html=True)
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=data.index, y=before_s,
                                        line=dict(color="#3b82f6", width=1.2), name="Original"))
            outlier_idx = data.index[mask]
            outlier_val = before_s[mask]
            fig_ts.add_trace(go.Scatter(x=outlier_idx, y=outlier_val, mode="markers", name="Outlier",
                                        marker=dict(color="#ef4444", size=8, symbol="x",
                                                    line=dict(color="#ef4444", width=2))))
            fig_ts.add_trace(go.Scatter(x=data.index, y=after_s,
                                        line=dict(color="#10b981", width=1.2, dash="dot"), name="After Treatment"))
            fig_ts.update_layout(**PLY, xaxis=dict(**AX), yaxis=dict(**AX), height=250,
                                  margin=dict(t=10,b=10,l=10,r=10),
                                  legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
            st.plotly_chart(fig_ts, use_container_width=True)

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Mean (Before)", f"{info['before_mean']:.4f}")
            sc2.metric("Mean (After)",  f"{info['after_mean']:.4f}",
                       delta=f"{info['after_mean']-info['before_mean']:+.4f}")
            sc3.metric("Std (Before)", f"{info['before_std']:.4f}")
            sc4.metric("Std (After)",  f"{info['after_std']:.4f}",
                       delta=f"{info['after_std']-info['before_std']:+.4f}")

            st.markdown('<p class="sec-title" style="margin-top:8px;">Q-Q Plot — Normality After Treatment</p>',
                        unsafe_allow_html=True)
            (osm, osr), (slope, intercept, r) = stats.probplot(after_s.dropna())
            qq_line = slope * np.array(osm) + intercept
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=list(osm), y=list(osr), mode="markers",
                                        marker=dict(color="#3b82f6", size=3, opacity=0.6), name="Sample"))
            fig_qq.add_trace(go.Scatter(x=list(osm), y=list(qq_line),
                                        line=dict(color="#ef4444", width=2), name=f"Normal line (R={r:.3f})"))
            fig_qq.update_layout(**PLY, xaxis=dict(title="Theoretical Quantiles", **AX),
                                  yaxis=dict(title="Sample Quantiles", **AX),
                                  height=260, margin=dict(t=10,b=10,l=10,r=10),
                                  legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
            st.plotly_chart(fig_qq, use_container_width=True)
        else:
            st.success("✅ All features are clean — no outliers detected or treated.")

        st.markdown('<p class="sec-title">Skewness Before vs After Treatment</p>',
                    unsafe_allow_html=True)
        skew_before = feat_df_raw[avail].skew().sort_values()
        skew_after  = feat_df_treated[avail].skew()
        fig_skew = go.Figure()
        fig_skew.add_trace(go.Bar(x=skew_before.index, y=skew_before.values,
                                   name="Before", marker_color="#ef4444", opacity=0.75))
        fig_skew.add_trace(go.Bar(x=skew_before.index,
                                   y=[skew_after[f] for f in skew_before.index],
                                   name="After", marker_color="#10b981", opacity=0.75))
        fig_skew.add_hline(y=0, line_color="#64748b", line_dash="dot")
        fig_skew.update_layout(**PLY, barmode="group",
                                xaxis=dict(tickangle=-35, **AX), yaxis=dict(title="Skewness", **AX),
                                height=320, margin=dict(t=10,b=80,l=10,r=10),
                                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
        st.plotly_chart(fig_skew, use_container_width=True)


    # ══════════════════════════════════════════════════════════
    #  TAB 3 — MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════
    with tab3:
        colors = [MODEL_META[n]["color"] for n in model_names]

        st.markdown('<p class="sec-title">Accuracy · ROC-AUC · F1 Score</p>', unsafe_allow_html=True)
        fig_bars = make_subplots(rows=1, cols=3,
                                 subplot_titles=["Accuracy","ROC-AUC","F1 Score"])
        for vals, ci in [
            ([perf[n]["accuracy"] for n in model_names], 1),
            ([perf[n]["auc"]      for n in model_names], 2),
            ([perf[n]["f1"]       for n in model_names], 3),
        ]:
            fig_bars.add_trace(go.Bar(
                x=model_names, y=vals, marker_color=colors,
                text=[f"{v:.3f}" for v in vals], textposition="outside", showlegend=False,
            ), row=1, col=ci)
        fig_bars.update_layout(
            **PLY, height=300, margin=dict(t=40,b=10,l=10,r=10),
            yaxis=dict(range=[0.3,0.85],**AX), yaxis2=dict(range=[0.3,0.85],**AX),
            yaxis3=dict(range=[0.3,0.85],**AX),
            xaxis=dict(**AX), xaxis2=dict(**AX), xaxis3=dict(**AX),
        )
        for ann in fig_bars.layout.annotations:
            ann.font.color = "#5b8fc9"; ann.font.size = 10
        st.plotly_chart(fig_bars, use_container_width=True)

        cols_cards = st.columns(3)
        for col, name in zip(cols_cards, model_names):
            p = perf[name]; mc = MODEL_META[name]["color"]
            col.markdown(f"""
            <div style="background:#0d1220;border:1px solid #1a2744;border-radius:10px;
                        padding:16px;margin-bottom:4px;">
              <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;
                          text-transform:uppercase;color:{mc};margin-bottom:12px;">{name}</div>
              <div class="perf-grid">
                <div class="perf-card"><div class="perf-val" style="color:{mc};">
                    {p['accuracy']:.3f}</div><div class="perf-lbl">Accuracy</div></div>
                <div class="perf-card"><div class="perf-val" style="color:{mc};">
                    {p['auc']:.3f}</div><div class="perf-lbl">AUC</div></div>
                <div class="perf-card"><div class="perf-val" style="color:{mc};">
                    {p['f1']:.3f}</div><div class="perf-lbl">F1</div></div>
              </div>
            </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="sec-title" style="margin-top:14px;">ROC Curves</p>',
                        unsafe_allow_html=True)
            fig_roc = go.Figure()
            fig_roc.add_shape(type="line",x0=0,x1=1,y0=0,y1=1,
                              line=dict(color="#1a2744",dash="dot"))
            for name in model_names:
                p = perf[name]
                fig_roc.add_trace(go.Scatter(
                    x=p["fpr"], y=p["tpr"],
                    name=f"{name} ({p['auc']:.3f})",
                    line=dict(color=MODEL_META[name]["color"],width=2),
                ))
            fig_roc.update_layout(
                **PLY, xaxis=dict(title="FPR",**AX), yaxis=dict(title="TPR",**AX),
                height=300, margin=dict(t=10,b=10,l=10,r=10),
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with c2:
            st.markdown('<p class="sec-title" style="margin-top:14px;">Confusion Matrix</p>',
                        unsafe_allow_html=True)
            cm_sel = st.selectbox("Model", model_names, key="cm_sel")
            cm = perf[cm_sel]["cm"]
            fig_cm = px.imshow(
                cm, text_auto=True, x=["Pred DOWN","Pred UP"],
                y=["Actual DOWN","Actual UP"],
                color_continuous_scale=[[0,"#080c14"],[0.5,"#1a2744"],
                                         [1,MODEL_META[cm_sel]["color"]]],
                aspect="auto",
            )
            fig_cm.update_layout(**PLY, height=300, margin=dict(t=10,b=10,l=10,r=10),
                                  coloraxis_showscale=False)
            fig_cm.update_traces(textfont_size=15)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown('<p class="sec-title">Random Forest — Feature Importances</p>',
                    unsafe_allow_html=True)
        fi   = perf["Random Forest"]["fi"]
        fi_s = pd.Series(fi, index=avail).sort_values()
        bar_colors = ["#10b981" if v >= fi_s.median() else "#3b82f6" for v in fi_s.values]
        fig_fi = go.Figure(go.Bar(
            x=fi_s.values, y=fi_s.index, orientation="h", marker_color=bar_colors,
            text=[f"{v:.4f}" for v in fi_s.values], textposition="outside",
            textfont=dict(size=9,color="#64748b"),
        ))
        fig_fi.update_layout(
            **PLY, xaxis=dict(title="Importance",**AX), yaxis=dict(**AX),
            height=400, margin=dict(t=10,b=10,l=10,r=60),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown('<p class="sec-title">Predicted Probability Distribution on Test Set</p>',
                    unsafe_allow_html=True)
        fig_prob = go.Figure()
        for name in model_names:
            fig_prob.add_trace(go.Violin(
                y=perf[name]["proba"], name=name,
                line_color=MODEL_META[name]["color"],
                fillcolor=_hex_to_rgba(MODEL_META[name]["color"], alpha=0.13),
                box_visible=True, meanline_visible=True,
            ))
        fig_prob.update_layout(
            **PLY, xaxis=dict(**AX), yaxis=dict(title="P(UP)",**AX),
            height=280, margin=dict(t=10,b=10,l=10,r=10),
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)), violinmode="group",
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown('<p class="sec-title">Summary Table</p>', unsafe_allow_html=True)
        summ = pd.DataFrame({
            "Model":    model_names,
            "Accuracy": [f"{perf[n]['accuracy']:.4f}" for n in model_names],
            "ROC-AUC":  [f"{perf[n]['auc']:.4f}"      for n in model_names],
            "F1 Score": [f"{perf[n]['f1']:.4f}"        for n in model_names],
            "Best?":    ["✅ Best AUC" if n == best_model else "" for n in model_names],
        }).set_index("Model")
        st.dataframe(summ, use_container_width=True)


    # ══════════════════════════════════════════════════════════
    #  TAB 4 — PREDICTION
    # ══════════════════════════════════════════════════════════
    with tab4:
        data_label = "Demo data" if is_syn else f"Live · {start_date} → {end_date}"

        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    background:#0d1220;border:1px solid #1a2744;border-radius:10px;
                    padding:14px 22px;margin-bottom:18px;">
            <div>
                <span style="font-family:'Space Mono',monospace;font-size:1.2rem;
                             font-weight:700;color:#e2e8f0;">{ticker_disp}</span>
                <span style="font-size:0.72rem;color:#334155;margin-left:10px;">{data_label}</span>
                <span style="font-size:0.68rem;color:#3b82f6;margin-left:12px;font-weight:600;">
                    Best model: {best_model} (AUC {perf[best_model]['auc']:.3f})
                </span>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.62rem;color:#3b82f6;font-weight:600;
                            letter-spacing:0.1em;text-transform:uppercase;">Last Close</div>
                <div style="font-family:'Space Mono',monospace;font-size:1.05rem;color:#e2e8f0;">
                    ${last_close:,.2f}
                    <span style="font-size:0.7rem;color:#334155;margin-left:6px;">{last_date}</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        if model_choice == "Ensemble (All 3)":
            cols_pred = st.columns(3)
            for col, name in zip(cols_pred, model_names):
                with col:
                    ap = all_preds[name]
                    render_result_card(name, ap["direction"], ap["prob"], ap["conf"])

            cons_cls  = "consensus-up" if consensus == "UP" else "consensus-down"
            cons_color2 = "#10b981" if consensus == "UP" else "#ef4444"
            arrow     = "▲" if consensus == "UP" else "▼"
            st.markdown(f"""
            <div class="{cons_cls}">
                <p style="font-size:0.62rem;font-weight:700;letter-spacing:0.15em;
                          text-transform:uppercase;color:#64748b;margin:0 0 6px;">
                    Ensemble Consensus
                </p>
                <p class="consensus-label" style="color:{cons_color2};">{arrow} {consensus}</p>
                <p class="consensus-sub">
                    {up_votes}/3 models predict UP &nbsp;·&nbsp; Avg P(Up) = {avg_prob:.3f}
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            _, center, _ = st.columns([1,2,1])
            with center:
                ap = all_preds[model_choice]
                render_result_card(model_choice, ap["direction"], ap["prob"], ap["conf"])

        st.markdown('<p class="sec-title" style="margin-top:24px;">P(UP) Gauge — All Models</p>',
                    unsafe_allow_html=True)
        gauge_vals   = [all_preds[n]["prob"] for n in model_names]
        gauge_colors = [MODEL_META[n]["color"] for n in model_names]
        fig_gauge = go.Figure(go.Bar(
            x=gauge_vals, y=model_names, orientation="h", marker_color=gauge_colors,
            text=[f"P(UP) = {v:.3f}" for v in gauge_vals], textposition="inside",
        ))
        fig_gauge.add_vline(x=0.5, line_color="#ffffff", line_dash="dot",
                            annotation_text="Decision boundary",
                            annotation_font_color="#64748b", annotation_position="top right")
        fig_gauge.update_layout(
            **PLY, xaxis=dict(range=[0,1],title="Probability",**AX), yaxis=dict(**AX),
            height=180, margin=dict(t=30,b=10,l=10,r=10), showlegend=False,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown('<p class="sec-title">All Models — Consensus Table</p>', unsafe_allow_html=True)
        rows = []
        for n in model_names:
            ap = all_preds[n]
            rows.append({
                "Model":      n,
                "P(UP)":      round(ap["prob"],4),
                "Direction":  "▲ UP" if ap["direction"]=="UP" else "▼ DOWN",
                "Confidence": f"{ap['conf']*100:.1f}%",
                "Test AUC":   round(perf[n]["auc"],4),
                "Test Acc":   round(perf[n]["accuracy"],4),
            })
        cons_df = pd.DataFrame(rows).set_index("Model")

        def style_dir(val):
            if "UP"   in str(val): return "color:#10b981; font-weight:700;"
            if "DOWN" in str(val): return "color:#ef4444; font-weight:700;"
            return ""

        st.dataframe(cons_df.style.map(style_dir, subset=["Direction"]),
                     use_container_width=True)

        st.markdown('<p class="sec-title" style="margin-top:20px;">Historical Prediction Accuracy on Test Set</p>',
                    unsafe_allow_html=True)
        test_len  = len(perf["Logistic Regression"]["y_truth"])
        test_idx  = data.index[-test_len:]
        actual_cl = data["Close"].iloc[-test_len:].values

        fig_hist_pred = go.Figure()
        fig_hist_pred.add_trace(go.Scatter(
            x=test_idx, y=actual_cl, line=dict(color="#64748b",width=1.2), name="Actual Close"
        ))
        for name in model_names:
            preds = perf[name]["preds"]
            mc    = MODEL_META[name]["color"]
            correct_up_idx = [i for i,(p,t) in enumerate(
                zip(preds, perf[name]["y_truth"])) if p==1 and t==1]
            if correct_up_idx:
                fig_hist_pred.add_trace(go.Scatter(
                    x=test_idx[correct_up_idx], y=actual_cl[correct_up_idx],
                    mode="markers", name=f"{MODEL_META[name]['short']} correct ▲",
                    marker=dict(color=mc,size=5,symbol="triangle-up"), opacity=0.7,
                ))
        fig_hist_pred.update_layout(
            **PLY, xaxis=dict(**AX), yaxis=dict(**AX), height=280,
            margin=dict(t=10,b=10,l=10,r=10),
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)),
        )
        st.plotly_chart(fig_hist_pred, use_container_width=True)