from Functions import *
from ExtraResults import run_all_extra_results

import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
processed_dir = os.path.join("C:/Users/juanz/Desktop/Universidad/ECG_HRV_Features/Processed_Data")
results_dir = os.path.join(ROOT_DIR, "Results")

fs = 500
lead_index = 1  # 0 = I, 1 = II, 2 = aVR

features = ["RR_mean", "SDRR", "RMSSD", "NN50", "pNN50"]

DATASETS = {
    1: "D1",
    3: "D3"
}

# ============================================================
# LOAD SIGNALS + LABELS
# ============================================================

def load_signals_labels_dataset(processed_dir=processed_dir, dataset=1):
    X = np.load(
        os.path.join(processed_dir, f"signals{dataset}.npy")
    ).astype(np.float32)

    y = np.load(
        os.path.join(processed_dir, f"labels{dataset}.npy")
    ).astype(int)

    return X, y


# ============================================================
# CREAR CARPETAS
# ============================================================

os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, "R_peaks_examples"), exist_ok=True)


# ============================================================
# VALIDACIÓN RR / BPM
# ============================================================

def validate_rr_bpm(df, dataset_name):
    print(f"\n==================== {dataset_name} | RRmean / BPM VALIDATION ====================")

    rows = []

    for cls in sorted(df["label"].unique()):
        vals = df[df["label"] == cls]["RR_mean"].dropna().values

        if len(vals) == 0:
            continue

        rr_median = np.median(vals)
        rr_mean = np.mean(vals)

        rows.append({
            "class": CLASS_NAMES[int(cls)],
            "RR_median_s": rr_median,
            "RR_mean_s": rr_mean,
            "BPM_from_median_RR": 60.0 / rr_median,
            "BPM_from_mean_RR": 60.0 / rr_mean,
            "valid_samples": len(vals)
        })

    out = pd.DataFrame(rows)

    print(out)

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

    X, y = load_signals_labels_dataset(
        processed_dir=processed_dir,
        dataset=dataset_id
    )

    check_dataset(X, y, name=dataset_name)

    # --------------------------------------------------------
    # R-PEAK VISUAL VALIDATION
    # --------------------------------------------------------

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

    df = build_dataframe(
        X=X,
        y=y,
        fs=fs,
        lead_index=lead_index,
        dataset_name=dataset_name
    )

    print(f"\n===== {dataset_name} FEATURE TABLE PREVIEW =====")
    print(df.head())
    print(f"\n===== {dataset_name} CLASS DISTRIBUTION =====")
    print(pd.Series(y).value_counts())

    # --------------------------------------------------------
    # RR / BPM VALIDATION
    # --------------------------------------------------------

    validate_rr_bpm(df, dataset_name)

    # --------------------------------------------------------
    # KRUSKAL-WALLIS
    # --------------------------------------------------------

    stats = run_stats(
        df=df,
        features=features
    )

    print(f"\n===== {dataset_name} KRUSKAL-WALLIS =====")
    print(stats)

    # --------------------------------------------------------
    # PAIRWISE MANN-WHITNEY + BONFERRONI
    # --------------------------------------------------------

    pairwise = run_pairwise_bonferroni(
        df=df,
        features=features
    )

    print(f"\n===== {dataset_name} PAIRWISE MANN-WHITNEY + BONFERRONI =====")
    print(pairwise)

    # --------------------------------------------------------
    # ONE-VS-REST AUC
    # --------------------------------------------------------

    auc = auc_features(
        df=df,
        features=features
    )

    print(f"\n===== {dataset_name} AUC FEATURES =====")
    print(auc)

    # --------------------------------------------------------
    # ML MODELS
    # --------------------------------------------------------

    models = train_models(
        df=df,
        features=features
    )

    print(f"\n===== {dataset_name} MODEL RESULTS =====")
    print(models)

    # --------------------------------------------------------
    # MINIMAL FEATURE ANALYSIS
    # --------------------------------------------------------

    minimal_results = minimal(
        df=df,
        features=features
    )

    print(f"\n===== {dataset_name} MINIMAL FEATURE ANALYSIS =====")
    print(minimal_results)

    # --------------------------------------------------------
    # DNN FEATURE MODEL
    # --------------------------------------------------------

    dnn = evaluate_dnn_feature_model(
        df=df,
        features=features,
        results_dir=None,
        prefix=dataset_name
    )

    print(dnn)

    print(f"\n==================== {dataset_name} | ANALYSIS FINISHED ====================")

    return X, y, df


# ============================================================
# RUN ALL DATASETS
# ============================================================

dfs = {}

for dataset_id, dataset_name in DATASETS.items():
    X, y, df = analyze_dataset(
        dataset_id=dataset_id,
        dataset_name=dataset_name
    )

    dfs[dataset_name] = df


# ============================================================
# COMPARISONS
# ============================================================

if len(dfs) >= 2:

    dataset_names = list(dfs.keys())

    df_all = pd.concat(
        [dfs[name] for name in dataset_names],
        axis=0,
        ignore_index=True
    )

    # Get all pairs of datasets for comparison
    if len(dfs) >= 2:
        dataset_list = list(dfs.items())
        
        # Process first two datasets (D1 vs D2 or D1 vs D3, etc.)
        dataset_name_1, df1 = dataset_list[0]
        dataset_name_2, df2 = dataset_list[1]

        shift_results = dataset_shift_tests(
            df1=df1,
            df2=df2,
            features=features,
            results_dir=None
        )

        print(f"\n===== {dataset_name_1} vs {dataset_name_2} FEATURE SHIFT TESTS =====")
        print(shift_results)

        plot_combined_boxplots(
            df1=df1,
            df2=df2,
            features=features,
            results_dir=f"{results_dir}/comparison/figures"
        )

        run_all_extra_results(
            df1=df1,
            df2=df2,
            base_dir=f"{results_dir}/extra_results",
            name1=dataset_name_1,
            name2=dataset_name_2
        )

print("\n\n==================== FULL ANALYSIS FINISHED ====================")
print("All results saved in the Results/ folder.")