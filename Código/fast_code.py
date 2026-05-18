from Functions import *

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

processed_dir = "Processed_Data"
results_dir = "Results/fast_analysis"

dataset_id = 1      # cambia a 2 para D2
fs = 500
lead_index = 1      # 0 = I, 1 = II, 2 = aVR

features = ["RR_mean", "SDRR", "RMSSD", "NN50", "pNN50"]

os.makedirs(results_dir, exist_ok=True)

# ============================================================
# R-PEAK DETECTION THRESHOLD LOGIC
# ============================================================

def detect_r_threshold(ecg, fs=500):
    ecg = preprocess_ecg_for_hrv(
        ecg,
        use_wavelet=True,
        normalize=True
    )

    upper_threshold = 0.60
    lower_threshold = 0.45

    ignore_reading = False
    peaks = []

    for i, value in enumerate(ecg):
        if value >= upper_threshold and not ignore_reading:
            peaks.append(i)
            ignore_reading = True

        if value < lower_threshold:
            ignore_reading = False

    min_distance = int(0.25 * fs)

    filtered_peaks = []
    last_peak = -999999

    for p in peaks:
        if p - last_peak >= min_distance:
            filtered_peaks.append(p)
            last_peak = p

    return np.array(filtered_peaks, dtype=int)

# ============================================================
# HRV FEATURE EXTRACTION
# ============================================================

def compute_hrv_threshold(ecg, fs=500):
    peaks = detect_r_threshold(ecg, fs=fs)

    if len(peaks) < 2:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, len(peaks)]

    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.30) & (rr < 2.50)]

    if len(rr) < 1:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, len(peaks)]

    rr_diff = np.abs(np.diff(rr))

    RR_mean = np.mean(rr)
    SDRR = np.std(rr, ddof=1) if len(rr) > 1 else 0.0
    RMSSD = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0.0
    NN50 = np.sum(rr_diff > 0.05)
    pNN50 = (NN50 / len(rr_diff)) * 100 if len(rr_diff) > 0 else 0.0

    return [RR_mean, SDRR, RMSSD, NN50, pNN50, len(peaks)]

# ============================================================
# LOAD DATA
# ============================================================

X, y = load_dataset_all(
    processed_dir=processed_dir,
    dataset=dataset_id
)

print("\n====================")
print(f"FAST ANALYSIS D{dataset_id}")
print("====================")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# BUILD DEBUG DATAFRAME
# ============================================================

rows = []

for idx in range(len(X)):
    ecg = X[idx, :, lead_index]
    vals = compute_hrv_threshold(ecg, fs=fs)

    rows.append({
        "id": idx,
        "label": int(y[idx]),
        "class": CLASS_NAMES[int(y[idx])],
        "RR_mean": vals[0],
        "SDRR": vals[1],
        "RMSSD": vals[2],
        "NN50": vals[3],
        "pNN50": vals[4],
        "n_peaks": vals[5],
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(results_dir, f"D{dataset_id}_features_fast.csv"), index=False)

# ============================================================
# VALID COUNTS
# ============================================================

print("\n====================")
print("VALID COUNTS")
print("====================")

valid_counts = df.groupby("class")[features].count()
print(valid_counts)
valid_counts.to_csv(os.path.join(results_dir, f"D{dataset_id}_valid_counts.csv"))

# ============================================================
# RR / BPM SUMMARY
# ============================================================

print("\n====================")
print("RRmean / BPM SUMMARY")
print("====================")

rr_rows = []

for cls in sorted(df["label"].unique()):
    class_name = CLASS_NAMES[int(cls)]
    vals = df[df["label"] == cls]["RR_mean"].dropna().values

    if len(vals) == 0:
        continue

    rr_med = np.median(vals)
    rr_mean = np.mean(vals)

    rr_rows.append({
        "class": class_name,
        "valid_samples": len(vals),
        "RR_median_s": rr_med,
        "RR_mean_s": rr_mean,
        "BPM_from_median_RR": 60.0 / rr_med,
        "BPM_from_mean_RR": 60.0 / rr_mean,
    })

rr_summary = pd.DataFrame(rr_rows)
print(rr_summary)
rr_summary.to_csv(os.path.join(results_dir, f"D{dataset_id}_rr_bpm_summary.csv"), index=False)

# ============================================================
# FULL FEATURE SUMMARY
# ============================================================

def q1(x):
    x = x.dropna()
    return np.percentile(x, 25) if len(x) > 0 else np.nan

def q3(x):
    x = x.dropna()
    return np.percentile(x, 75) if len(x) > 0 else np.nan

print("\n====================")
print("FULL FEATURE SUMMARY")
print("====================")

summary = (
    df.groupby("class")[features]
    .agg(["count", "mean", "median", "std", "min", "max", q1, q3])
)

summary.columns = [
    "_".join([str(c) for c in col])
    for col in summary.columns
]

print(summary)
summary.to_csv(os.path.join(results_dir, f"D{dataset_id}_feature_summary.csv"))

# ============================================================
# STANDARD DEVIATION ONLY
# ============================================================

print("\n====================")
print("STANDARD DEVIATION BY CLASS")
print("====================")

std_table = df.groupby("class")[features].std()
print(std_table)
std_table.to_csv(os.path.join(results_dir, f"D{dataset_id}_std_by_class.csv"))

# ============================================================
# DISCRETIZATION CHECK
# ============================================================

print("\n====================")
print("UNIQUE VALUES CHECK FOR NN50 / pNN50")
print("====================")

for f in ["NN50", "pNN50"]:
    print(f"\n--- {f} ---")
    for cls in sorted(df["label"].unique()):
        class_name = CLASS_NAMES[int(cls)]
        vals = df[df["label"] == cls][f].dropna().values
        unique_vals, counts = np.unique(vals, return_counts=True)

        temp = pd.DataFrame({
            f: unique_vals,
            "count": counts,
            "percentage": counts / counts.sum() * 100
        })

        print(f"\nClass {class_name}")
        print(temp.head(15))

        temp.to_csv(
            os.path.join(results_dir, f"D{dataset_id}_{class_name}_{f}_unique_values.csv"),
            index=False
        )

# ============================================================
# N PEAKS DISTRIBUTION
# ============================================================

print("\n====================")
print("NUMBER OF DETECTED R-PEAKS")
print("====================")

peaks_summary = (
    df.groupby("class")["n_peaks"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(peaks_summary)
peaks_summary.to_csv(os.path.join(results_dir, f"D{dataset_id}_n_peaks_summary.csv"))

# ============================================================
# BOXPLOTS
# ============================================================

print("\n====================")
print("SAVING BOXPLOTS")
print("====================")

for f in features:
    plt.figure(figsize=(8, 5))

    data = []
    labels = []

    for cls in sorted(df["label"].unique()):
        vals = df[df["label"] == cls][f].dropna().values
        data.append(vals)
        labels.append(CLASS_NAMES[int(cls)])

    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(f"D{dataset_id} - {f}")
    plt.xlabel("Rhythm class")
    plt.ylabel(f)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(results_dir, f"D{dataset_id}_{f}_boxplot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)

# ============================================================
# SAMPLE R-PEAK PLOTS
# ============================================================

print("\n====================")
print("SAVING R-PEAK EXAMPLES")
print("====================")

for cls in sorted(np.unique(y)):
    idxs = np.where(y == cls)[0]
    class_name = CLASS_NAMES[int(cls)]

    for k, idx in enumerate(idxs[:3]):
        ecg_raw = X[idx, :, lead_index]

        ecg = preprocess_ecg_for_hrv(
            ecg_raw,
            use_wavelet=True,
            normalize=True
        )

        peaks = detect_r_threshold(ecg_raw, fs=fs)

        rr_mean, sdr, rmssd, nn50, pnn50, n_peaks = compute_hrv_threshold(ecg_raw, fs=fs)

        t = np.arange(len(ecg)) / fs

        plt.figure(figsize=(12, 4))
        plt.plot(t, ecg, linewidth=1)

        if len(peaks) > 0:
            plt.scatter(t[peaks], ecg[peaks], s=30)

        bpm = 60.0 / rr_mean if np.isfinite(rr_mean) and rr_mean > 0 else np.nan

        plt.title(
            f"D{dataset_id} | {class_name} | "
            f"RR={rr_mean:.3f}s | BPM={bpm:.1f} | peaks={n_peaks}"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized ECG")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(results_dir, f"D{dataset_id}_{class_name}_sample{k}_rpeaks.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print("Saved:", save_path)

print("\n====================")
print("FAST ANALYSIS FINISHED")
print("====================")
print("Saved in:", results_dir)

# ============================================================
# RMSSD / SDRR / RRmean : MEAN +- STD
# FOR D1 AND D2
# ============================================================
