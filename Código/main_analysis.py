from Functions import *
from ExtraResults import run_all_extra_results

import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

processed_dir = "Processed_Data"
results_dir = "Results"

fs = 500
lead_index = 1  # 0 = I, 1 = II, 2 = aVR

features = ["RR_mean", "SDRR", "RMSSD", "NN50", "pNN50"]

# ============================================================
# CREAR CARPETAS
# ============================================================

folders = [
    "Results/D1/tables",
    "Results/D1/figures",
    "Results/D2/tables",
    "Results/D2/figures",
    "Results/comparison/tables",
    "Results/comparison/figures",
    "Results/R_peaks_examples",
    "Results/extra_results",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# VALIDACIÓN RR / BPM
# ============================================================

def validate_rr_bpm(df, dataset_name):
    print(f"\n==================== {dataset_name} | RRmean / BPM VALIDATION ====================")
    print("Expected physiological order: ST = lower RR / higher BPM, SR = intermediate, SB = higher RR / lower BPM.")

    rows = []

    for cls in sorted(df["label"].unique()):
        vals = df[df["label"] == cls]["RR_mean"].dropna().values

        if len(vals) == 0:
            continue

        rr_median = np.median(vals)
        rr_mean = np.mean(vals)
        bpm_median = 60.0 / rr_median
        bpm_mean = 60.0 / rr_mean

        rows.append({
            "class": CLASS_NAMES[int(cls)],
            "RR_median_s": rr_median,
            "RR_mean_s": rr_mean,
            "BPM_from_median_RR": bpm_median,
            "BPM_from_mean_RR": bpm_mean,
            "valid_samples": len(vals)
        })

    out = pd.DataFrame(rows)
    print(out)

    out.to_csv(
        f"Results/{dataset_name}/tables/rr_bpm_validation.csv",
        index=False
    )

    return out

# ============================================================
# ANALIZAR DATASET
# ============================================================

def analyze_dataset(dataset_id, dataset_name):
    print(f"\n\n============================================================")
    print(f"STARTING ANALYSIS FOR {dataset_name}")
    print(f"============================================================")

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------
    print(f"\n==================== {dataset_name} | LOADING DATA ====================")

    X, y = load_dataset_all(
        processed_dir=processed_dir,
        dataset=dataset_id
    )

    check_dataset(X, y, name=dataset_name)

    # --------------------------------------------------------
    # R-PEAK VISUAL VALIDATION
    # --------------------------------------------------------
    print(f"\n==================== {dataset_name} | R-PEAK VISUAL VALIDATION ====================")
    print("Generates example ECG plots with detected R-peaks for each rhythm class.")

    plot_r_peak_examples(
        X=X,
        y=y,
        fs=fs,
        dataset_name=dataset_name,
        lead_index=lead_index,
        results_dir=os.path.join(results_dir, "R_peaks_examples"),
        samples_per_class=3
    )

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------
    print(f"\n==================== {dataset_name} | HRV FEATURE EXTRACTION ====================")
    print("Extracts RR_mean, SDRR, RMSSD, NN50, and pNN50 from lead II.")

    df = build_dataframe(
        X=X,
        y=y,
        fs=fs,
        lead_index=lead_index,
        dataset_name=dataset_name
    )

    df.to_csv(
        f"Results/{dataset_name}/tables/features.csv",
        index=False
    )

    pd.Series(y).value_counts().to_csv(
        f"Results/{dataset_name}/tables/class_distribution.csv"
    )

    print("Feature table saved.")
    print(df.head())

    # --------------------------------------------------------
    # RR / BPM VALIDATION
    # --------------------------------------------------------
    validate_rr_bpm(df, dataset_name)

    # --------------------------------------------------------
    # KRUSKAL-WALLIS
    # --------------------------------------------------------
    print(f"\n==================== {dataset_name} | KRUSKAL-WALLIS TEST ====================")
    print("Evaluates global statistical differences among rhythm classes for each HRV feature.")

    stats = run_stats(
        df=df,
        features=features,
        save_path=f"Results/{dataset_name}/tables/stats.csv"
    )

    print(stats)

    # --------------------------------------------------------
    # PAIRWISE MANN-WHITNEY + BONFERRONI
    # --------------------------------------------------------
    print(f"\n==================== {dataset_name} | MANN-WHITNEY U + BONFERRONI ====================")
    print("Evaluates pairwise differences between rhythm classes for each HRV feature.")

    pairwise = run_pairwise_bonferroni(
        df=df,
        features=features,
        save_path=f"Results/{dataset_name}/tables/pairwise_bonferroni.csv"
    )

    print(pairwise.head(30))

    # --------------------------------------------------------
    # ONE-VS-REST AUC
    # --------------------------------------------------------
    print(f"\n==================== {dataset_name} | ONE-VS-REST AUC ====================")
    print("Evaluates the univariate discriminative capability of each HRV feature for each rhythm class.")

    auc = auc_features(
        df=df,
        features=features,
        output_path=f"Results/{dataset_name}/tables/auc_features.csv"
    )

    print(auc)

    # --------------------------------------------------------
    # ML MODELS
    # --------------------------------------------------------
    print(f"\n==================== {dataset_name} | ML MODEL PERFORMANCE ====================")
    print("Evaluates Logistic Regression and Random Forest using 5-fold cross-validation.")

    models = train_models(
        df=df,
        features=features,
        output_path=f"Results/{dataset_name}/tables/model_results.csv"
    )

    print(models)

    # --------------------------------------------------------
    # MINIMAL FEATURE ANALYSIS
    # --------------------------------------------------------
    print(f"\n==================== {dataset_name} | MINIMAL FEATURE ANALYSIS ====================")
    print("Evaluates how macro F1-score changes as HRV features are progressively added.")

    minimal_results = minimal(
        df=df,
        features=features,
        output_path=f"Results/{dataset_name}/tables/minimal_features.csv"
    )

    print(minimal_results)

    # --------------------------------------------------------
    # BOXPLOTS PER DATASET
    # --------------------------------------------------------
    print(f"\n==================== {dataset_name} | BOXPLOTS ====================")
    print("Generates boxplots for visual inspection of HRV feature distributions across rhythm classes.")

    boxplots(
        df=df,
        save_dir=f"Results/{dataset_name}/figures",
        features=features
    )

    # --------------------------------------------------------
    # DNN FEATURE MODEL
    # --------------------------------------------------------
    print(f"\n==================== {dataset_name} | DNN FEATURE MODEL ====================")
    print("Evaluates a simple neural network using only extracted HRV features.")

    dnn = evaluate_dnn_feature_model(
        df=df,
        features=features,
        results_dir=f"Results/{dataset_name}/tables",
        prefix=dataset_name
    )

    print(dnn)

    print(f"\n==================== {dataset_name} | ANALYSIS FINISHED ====================")

    return X, y, df

# ============================================================
# RUN D1 AND D2
# ============================================================

X1, y1, df1 = analyze_dataset(
    dataset_id=1,
    dataset_name="D1"
)

X2, y2, df2 = analyze_dataset(
    dataset_id=2,
    dataset_name="D2"
)

# ============================================================
# COMPARISON D1 VS D2
# ============================================================

print("\n\n============================================================")
print("STARTING INTER-DATASET COMPARISON")
print("============================================================")

df_all = pd.concat([df1, df2], axis=0, ignore_index=True)

df_all.to_csv(
    "Results/comparison/tables/features_D1_D2.csv",
    index=False
)

# ------------------------------------------------------------
# FEATURE SHIFT TESTS
# ------------------------------------------------------------
print("\n==================== INTER-DATASET FEATURE SHIFT ANALYSIS ====================")
print("Compares D1 and D2 feature distributions within each rhythm class using MW/KS tests.")

shift_results = dataset_shift_tests(
    df1=df1,
    df2=df2,
    features=features,
    results_dir="Results/comparison/tables"
)

print(shift_results)

# ------------------------------------------------------------
# COMBINED BOXPLOTS
# ------------------------------------------------------------
print("\n==================== COMBINED D1-D2 BOXPLOTS ====================")
print("Generates combined boxplots to visually compare HRV feature distributions between datasets.")

plot_combined_boxplots(
    df1=df1,
    df2=df2,
    features=features,
    results_dir="Results/comparison/figures"
)

# ------------------------------------------------------------
# EXTRA RESULTS
# ------------------------------------------------------------
print("\n==================== EXTRA RESULTS FOR PAPER ====================")
print("Runs additional analyses such as feature ranking, confusion metrics, and paper-ready summaries.")

run_all_extra_results(
    df1=df1,
    df2=df2,
    base_dir="Results/extra_results"
)

print("\n\n==================== FULL ANALYSIS FINISHED ====================")
print("All results saved in the Results/ folder.")