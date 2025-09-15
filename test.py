import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import textwrap
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------
# CONFIG
# -----------------------
BASE = Path(r"E:\Code\semester 5\visualisasi dan intepretasi data\Mata Uang")
FILES = {
    "USDIDR": BASE / "USDIDR=X.csv",
    "MYRUSD": BASE / "MYRUSD=X.csv",
    "SGDUSD": BASE / "SGDUSD=X.csv",
    "THBUSD": BASE / "THBUSD=X.csv",
}
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------
# HELPERS
# -----------------------
def read_any_table(path: Path):
    """Read CSV or Excel-encoded file robustly."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        # Try CSV
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_excel(path)
        except Exception as e:
            raise RuntimeError(f"Unable to read {path}: {e}")

def basic_clean(df: pd.DataFrame):
    """Cleaning: drop exact duplicates, convert Date to datetime, sort."""
    # Drop full-row duplicates
    df = df.drop_duplicates().copy()
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Try find a date-like column
    date_cols = [c for c in df.columns if c.lower() in ("date", "timestamp", "time")]
    if not date_cols:
        # try heuristic
        for c in df.columns:
            if "date" in c.lower():
                date_cols.append(c)
    if not date_cols:
        raise RuntimeError("No date column detected in dataframe columns: " + ", ".join(df.columns))
    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors="coerce")
    df = df.sort_values(by=date_cols[0]).reset_index(drop=True)
    return df.rename(columns={date_cols[0]: "Date"})

def handle_missing(df: pd.DataFrame):
    """Fill missing numeric values with forward-fill then backward-fill."""
    # forward fill for time-series sensible fill
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].ffill().bfill()
    # If Date had NaT, drop those rows
    df = df[df["Date"].notna()].reset_index(drop=True)
    return df

def derive_features(df: pd.DataFrame, close_col_candidates=("Close","Adj Close","close","AdjClose")):
    """Select close column and derive returns and normalized close."""
    df = df.copy()
    # find close column name
    close_col = None
    for c in close_col_candidates:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        # try a numeric column that makes sense (last numeric)
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise RuntimeError("No numeric column for close price found.")
        close_col = num_cols[-1]

    df = df[["Date", close_col]].rename(columns={close_col: "Close"})
    # compute daily return (%) and absolute price
    df["Return"] = df["Close"].pct_change() * 100
    # normalization will be applied later per-series
    return df

# -----------------------
# 1) LOAD, CLEAN, TRANSFORM each file
# -----------------------
series_dict = {}
for key, path in FILES.items():
    print(f"Reading {key} from {path} ...")
    df_raw = read_any_table(path)
    df = basic_clean(df_raw)
    df = handle_missing(df)
    df = derive_features(df)
    # keep daily (we will resample later)
    series_dict[key] = df

# -----------------------
# 2) Transformation: Normalization (Min-Max) per series
# -----------------------
scalers = {}
for key, df in series_dict.items():
    scaler = MinMaxScaler()
    # fit on Close
    # avoid NaN in first row due to pct_change
    close_vals = df["Close"].fillna(method="ffill").values.reshape(-1,1)
    df["Close_norm"] = scaler.fit_transform(close_vals)
    scalers[key] = scaler
    # fill any NaN in Return (first row) with 0
    df["Return"] = df["Return"].fillna(0)
    series_dict[key] = df

# -----------------------
# 3) Reduction: resample to weekly and monthly aggregates
# -----------------------
def resample_agg(df, freq="W"):
    """Resample by Date index (first set index). use mean for Close & Return."""
    tmp = df.set_index("Date").resample(freq).mean().reset_index()
    return tmp

resampled_weekly = {}
resampled_monthly = {}
for key, df in series_dict.items():
    resampled_weekly[key] = resample_agg(df, freq="W")
    resampled_monthly[key] = resample_agg(df, freq="M")

# -----------------------
# 4) Integration: merge all monthly series on Date (inner join)
# -----------------------
# Build monthly merged df
dfs_to_merge = []
for key, df in resampled_monthly.items():
    # rename columns to include key
    tmp = df.rename(columns={"Close": f"Close_{key}", "Return": f"Return_{key}", "Close_norm": f"Close_norm_{key}"})
    dfs_to_merge.append(tmp[["Date", f"Close_{key}", f"Return_{key}", f"Close_norm_{key}"]])

from functools import reduce
df_merged = reduce(lambda left, right: pd.merge(left, right, on="Date", how="inner"), dfs_to_merge)

# Also create weekly merged for more granular plot if needed
dfs_weekly = []
for key, df in resampled_weekly.items():
    tmp = df.rename(columns={"Close": f"Close_{key}", "Return": f"Return_{key}", "Close_norm": f"Close_norm_{key}"})
    dfs_weekly.append(tmp[["Date", f"Close_{key}", f"Return_{key}", f"Close_norm_{key}"]])
df_merged_weekly = reduce(lambda left, right: pd.merge(left, right, on="Date", how="inner"), dfs_weekly)

# -----------------------
# 5) Basic statistics & relationships
# -----------------------
# Correlation matrix for returns
return_cols = [c for c in df_merged.columns if c.startswith("Return_")]
corr_returns = df_merged[return_cols].corr()

# Volatility (std of returns)
volatility = df_merged[return_cols].std().rename("Std_Return_%")

# Pairwise correlation for closes (raw)
close_cols = [c for c in df_merged.columns if c.startswith("Close_")]
corr_closes = df_merged[close_cols].corr()

# -----------------------
# 6) VISUALIZATIONS, save images and create PDF
# -----------------------
sns.set(style="whitegrid", font_scale=1.0)

# helper to save figure
def save_fig(fig, name):
    p = OUTPUT_DIR / name
    fig.savefig(p, bbox_inches="tight", dpi=150)
    print("Saved:", p)
    return p

pdf_path = OUTPUT_DIR / "report_preprocessing_fixed.pdf"
with PdfPages(pdf_path) as pdf:
    # --- Page 1: Title + summary (wrapped text) ---
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    title = "Preprocessing & EDA Report — Currency Series"
    subtitle = "Files: " + ", ".join([p.name for p in FILES.values()])
    summary_lines = (
        "- Dropped duplicates, coerced Date, forward/backfill missing numeric values.\n"
        "- Derived daily return (%) and normalized Close (MinMax).\n"
        "- Resampled to monthly averages and merged all series on Date.\n"
        "- Computed correlations and volatility for returns."
    )
    ax.text(0.01, 0.95, title, fontsize=20, weight="bold", va="top")
    ax.text(0.01, 0.90, subtitle, fontsize=10, va="top")
    wrapped = textwrap.fill(summary_lines, width=100)
    ax.text(0.01, 0.78, wrapped, fontsize=10, va="top")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 2: Monthly dual-axis (USD/IDR absolute & others normalized) ---
    fig, ax1 = plt.subplots(figsize=(11,6))
    ax1.plot(df_merged["Date"], df_merged["Close_USDIDR"], label="USD/IDR", color="tab:blue", linewidth=1.5)
    ax1.set_ylabel("USD/IDR", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df_merged["Date"], df_merged["Close_norm_MYRUSD"], label="MYR/USD (norm)", color="tab:orange", linestyle="--")
    ax2.plot(df_merged["Date"], df_merged["Close_norm_SGDUSD"], label="SGD/USD (norm)", color="tab:green", linestyle="--")
    ax2.plot(df_merged["Date"], df_merged["Close_norm_THBUSD"], label="THB/USD (norm)", color="tab:red", linestyle="--")
    ax2.set_ylabel("Normalized Close (0-1)", color="tab:gray")
    ax1.set_title("Monthly Average — USD/IDR (absolute) vs Other currencies (normalized)")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 3: Pairplot for returns (careful with layout) ---
    pairplot_df = df_merged[[c for c in df_merged.columns if c.startswith("Return_")]].dropna()
    if not pairplot_df.empty and pairplot_df.shape[1] > 1:
        g = sns.pairplot(pairplot_df, kind="reg", diag_kind="kde",
                        plot_kws={"scatter_kws": {"s": 20, "alpha": 0.6}})
        g.fig.suptitle("Pairwise Relationships — Returns (monthly)", y=1.02)
        g.fig.tight_layout()
        pdf.savefig(g.fig)
        plt.close(g.fig)

    # --- Page 4: Heatmap correlation ---
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr_returns, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation Matrix — Monthly Returns")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 5: Volatility bar plot ---
    fig, ax = plt.subplots(figsize=(8,5))
    volatility.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Volatility (Std of Monthly Returns)")
    ax.set_xlabel("Std(Return %)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Pages for each currency monthly Close ---
    for key in FILES.keys():
        fig, ax = plt.subplots(figsize=(11,5))
        col = f"Close_{key}"
        if col in df_merged.columns:
            ax.plot(df_merged["Date"], df_merged[col], marker='o', markersize=3, linewidth=1)
            ax.set_title(f"Monthly Average Close — {key}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Close")
            fig.tight_layout()
            pdf.savefig(fig)
        plt.close(fig)

    # --- Final page: Conclusions (wrapped) ---
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    conclusions_text = (
        "Conclusions:\n\n"
        "- USD/IDR shows high absolute values (because IDR has low per-unit value vs USD).\n"
        "- After normalization, MYR/SGD/THB display comparable dynamics that can be contrasted to USD/IDR.\n"
        f"- Pairwise correlation (returns):\n{corr_returns.round(2).to_string()}\n\n"
        f"- Volatility (std of monthly returns):\n{volatility.round(4).to_string()}\n\n"
        "- The processed datasets are ready for forecasting or risk analysis (ARIMA, LSTM, cointegration)."
    )
    wrapped2 = textwrap.fill(conclusions_text, width=100)
    ax.text(0.01, 0.98, "Preprocessing Report — Conclusions", fontsize=16, weight="bold", va="top")
    ax.text(0.01, 0.90, wrapped2, fontsize=10, va="top")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print("PDF generated at:", pdf_path.resolve())

# Optionally: print a short summary to terminal
print("\n-- SUMMARY STATS --")
print("Return correlations (monthly):\n", corr_returns.round(3))
print("\nVolatility (Std of returns):\n", volatility.round(4))