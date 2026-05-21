from Functions import *

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_EXTRA_RESULTS_DIR = os.path.join(ROOT_DIR, "Results", "extra_results")

FEATURES = ["RR_mean", "SDRR", "RMSSD", "NN50", "pNN50"]
CLASS_ORDER = [0, 1, 2, 3]
CLASS_LABELS = ["SR", "AF", "ST", "SB"]


# ============================================================
# UTILIDADES
# ============================================================

def make_dir(path):
    os.makedirs(path, exist_ok=True)


def clean_feature_matrix(df, features):
    X = df[features].replace([np.inf, -np.inf], np.nan).values
    y = df["label"].values.astype(int)
    return X, y


def cohens_d(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    x1 = x1[np.isfinite(x1)]
    x2 = x2[np.isfinite(x2)]

    n1, n2 = len(x1), len(x2)

    if n1 < 2 or n2 < 2:
        return np.nan

    s1 = np.var(x1, ddof=1)
    s2 = np.var(x2, ddof=1)

    pooled = np.sqrt(
        ((n1 - 1) * s1 + (n2 - 1) * s2) /
        (n1 + n2 - 2)
    )

    if pooled == 0:
        return np.nan

    return (np.mean(x1) - np.mean(x2)) / pooled


# ============================================================
# 1. EFFECT SIZE + POST-HOC
# ============================================================

def effect_size_pairwise(df, dataset_name, save_dir):
    rows = []
    labels = sorted(df["label"].unique())
    pairs = list(itertools.combinations(labels, 2))
    n_tests = len(pairs) * len(FEATURES)

    for feature in FEATURES:
        for c1, c2 in pairs:

            x1 = df[df["label"] == c1][feature].dropna().values
            x2 = df[df["label"] == c2][feature].dropna().values

            if len(x1) > 0 and len(x2) > 0:
                U, p = mannwhitneyu(x1, x2, alternative="two-sided")
            else:
                U, p = np.nan, np.nan

            p_bonf = min(p * n_tests, 1.0) if np.isfinite(p) else np.nan
            d = cohens_d(x1, x2)

            rows.append({
                "dataset": dataset_name,
                "feature": feature,
                "class_1": CLASS_NAMES[int(c1)],
                "class_2": CLASS_NAMES[int(c2)],
                "U": U,
                "p_raw": p,
                "p_bonferroni": p_bonf,
                "significant_bonferroni": p_bonf < 0.05 if np.isfinite(p_bonf) else False,
                "cohens_d": d,
                "abs_cohens_d": abs(d)
            })

    out = pd.DataFrame(rows)
    make_dir(save_dir)

    print(f"\n===== {dataset_name} PAIRWISE EFFECT SIZE =====")
    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.width', 200,
        'display.expand_frame_repr', False
    ):
        print(out)

    out.to_csv(os.path.join(save_dir, f"{dataset_name}_pairwise_effect_size.csv"), index=False)
    print(f"Pairwise effect size table saved en: {os.path.join(save_dir, f'{dataset_name}_pairwise_effect_size.csv')}")

    return out


# ============================================================
# 2. RANKING DE FEATURES POR AUC
# ============================================================

def auc_ranking(df, dataset_name, save_dir):
    make_dir(save_dir)

    rows = []
    y = df["label"].values.astype(int)

    for feature in FEATURES:
        x = df[feature].values.astype(float)

        med = np.nanmedian(x)
        x = np.nan_to_num(x, nan=med, posinf=med, neginf=med)

        aucs = []

        for cls in sorted(np.unique(y)):
            y_bin = (y == cls).astype(int)

            if len(np.unique(y_bin)) < 2:
                auc_val = np.nan
            else:
                auc_val = roc_auc_score(y_bin, x)
                auc_val = max(auc_val, 1 - auc_val)

            aucs.append(auc_val)

            rows.append({
                "dataset": dataset_name,
                "feature": feature,
                "class": CLASS_NAMES[int(cls)],
                "auc_ovr": auc_val
            })

        rows.append({
            "dataset": dataset_name,
            "feature": feature,
            "class": "mean",
            "auc_ovr": np.nanmean(aucs)
        })

    out = pd.DataFrame(rows)

    print(f"\n===== {dataset_name} AUC RANKING =====")
    print(out)

    rank = out[out["class"] == "mean"].sort_values("auc_ovr", ascending=False)

    plt.figure(figsize=(6, 4))
    plt.bar(rank["feature"], rank["auc_ovr"])
    plt.ylim(0.5, 1.0)
    plt.ylabel("Mean one-vs-rest AUC")
    plt.title(f"{dataset_name} - Feature AUC Ranking")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{dataset_name}_auc_ranking.png"),
        dpi=300
    )
    plt.close()

    return out


# ============================================================
# 3. MODELOS + MATRIZ DE CONFUSIÓN + ERROR ANALYSIS
# ============================================================

def model_error_analysis(df, dataset_name, save_dir):
    make_dir(save_dir)

    X, y = clean_feature_matrix(df, FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    models = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    summary = []

    for model_name, model in models.items():

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        summary.append({
            "dataset": dataset_name,
            "model": model_name,
            "accuracy": acc,
            "f1_macro": f1
        })

        report = classification_report(
            y_test,
            y_pred,
            labels=CLASS_ORDER,
            target_names=CLASS_LABELS,
            output_dict=True,
            zero_division=0
        )

        cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

        cm_df = pd.DataFrame(
            cm,
            index=CLASS_LABELS,
            columns=CLASS_LABELS
        )

        print(f"\n===== {dataset_name} {model_name} CLASSIFICATION REPORT =====")
        print(pd.DataFrame(report).transpose())
        print(f"\n===== {dataset_name} {model_name} CONFUSION MATRIX =====")
        print(cm_df)

        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.title(f"{dataset_name} - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(np.arange(len(CLASS_LABELS)), CLASS_LABELS)
        plt.yticks(np.arange(len(CLASS_LABELS)), CLASS_LABELS)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

        plt.colorbar()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                save_dir,
                f"{dataset_name}_{model_name}_confusion_matrix.png"
            ),
            dpi=300
        )
        plt.close()

        err = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred
        })

        err["true_class"] = err["y_true"].map(CLASS_NAMES)
        err["pred_class"] = err["y_pred"].map(CLASS_NAMES)

        err_table = (
            err[err["y_true"] != err["y_pred"]]
            .groupby(["true_class", "pred_class"])
            .size()
            .reset_index(name="n_errors")
            .sort_values("n_errors", ascending=False)
        )

        print(f"\n===== {dataset_name} {model_name} ERROR ANALYSIS =====")
        print(err_table)

    summary_df = pd.DataFrame(summary)
    print(f"\n===== {dataset_name} MODEL SUMMARY =====")
    print(summary_df)

    return summary_df


# ============================================================
# 4. FEATURE IMPORTANCE
# ============================================================

def feature_importance(df, dataset_name, save_dir):
    make_dir(save_dir)

    X, y = clean_feature_matrix(df, FEATURES)

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    rf.fit(X, y)

    rf_imp = rf.named_steps["clf"].feature_importances_

    rf_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": rf_imp
    }).sort_values("importance", ascending=False)

    print(f"\n===== {dataset_name} RF FEATURE IMPORTANCE =====")
    print(rf_df)

    plt.figure(figsize=(6, 4))
    plt.bar(rf_df["feature"], rf_df["importance"])
    plt.ylabel("Importance")
    plt.title(f"{dataset_name} - RF Feature Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{dataset_name}_RF_feature_importance.png"),
        dpi=300
    )
    plt.close()

    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    lr.fit(X, y)

    coefs = lr.named_steps["clf"].coef_

    lr_df = pd.DataFrame(
        coefs,
        columns=FEATURES,
        index=CLASS_LABELS
    )

    print(f"\n===== {dataset_name} LR COEFFICIENTS =====")
    print(lr_df)

    return rf_df, lr_df


# ============================================================
# 5. MINIMAL FEATURE ANALYSIS
# ============================================================

def minimal_feature_analysis(df, dataset_name, save_dir):
    make_dir(save_dir)

    rows = []

    for k in range(1, len(FEATURES) + 1):

        selected = FEATURES[:k]
        X, y = clean_feature_matrix(df, selected)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        models = {
            "LogisticRegression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42
                ))
            ]),

            "RandomForest": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                ))
            ])
        }

        for model_name, model in models.items():

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "n_features": k,
                "features_used": ", ".join(selected),
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            })

    out = pd.DataFrame(rows)

    print(f"\n===== {dataset_name} MINIMAL FEATURE ANALYSIS =====")
    print(out)

    plt.figure(figsize=(7, 5))

    for model_name in out["model"].unique():
        temp = out[out["model"] == model_name]
        plt.plot(
            temp["n_features"],
            temp["f1_macro"],
            marker="o",
            label=model_name
        )

    plt.xlabel("Number of features")
    plt.ylabel("Macro F1-score")
    plt.title(f"{dataset_name} - Minimal Feature Analysis")
    plt.xticks(range(1, len(FEATURES) + 1))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{dataset_name}_minimal_feature_analysis.png"),
        dpi=300
    )
    plt.close()

    return out


# ============================================================
# 6. HISTOGRAMAS
# ============================================================

def histograms_by_class(df, dataset_name, save_dir):
    hist_dir = save_dir
    make_dir(hist_dir)

    for feature in FEATURES:

        plt.figure(figsize=(7, 5))

        for cls in sorted(df["label"].unique()):
            vals = df[df["label"] == cls][feature].dropna().values
            plt.hist(vals, bins=40, alpha=0.4, label=CLASS_NAMES[int(cls)])

        plt.title(f"{dataset_name} - {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(hist_dir, f"{dataset_name}_{feature}_histogram.png"),
            dpi=300
        )
        plt.close()


# ============================================================
# 7. COMPARACIÓN D1 VS D2
# ============================================================

def compare_datasets(df1, df2, save_dir, name1="D1", name2="D2"):
    make_dir(save_dir)

    rows = []

    for feature in FEATURES:
        for cls in CLASS_ORDER:

            x1 = df1[df1["label"] == cls][feature].dropna().values
            x2 = df2[df2["label"] == cls][feature].dropna().values

            if len(x1) > 0 and len(x2) > 0:
                U, p = mannwhitneyu(x1, x2, alternative="two-sided")
                d = cohens_d(x1, x2)
            else:
                U, p, d = np.nan, np.nan, np.nan

            rows.append({
                "feature": feature,
                "class": CLASS_NAMES[int(cls)],
                f"{name1}_median": np.nanmedian(x1) if len(x1) else np.nan,
                f"{name2}_median": np.nanmedian(x2) if len(x2) else np.nan,
                f"{name1}_mean": np.nanmean(x1) if len(x1) else np.nan,
                f"{name2}_mean": np.nanmean(x2) if len(x2) else np.nan,
                "mannwhitney_U": U,
                "p_value": p,
                f"cohens_d_{name1}_vs_{name2}": d,
                "abs_cohens_d": abs(d)
            })

    out = pd.DataFrame(rows)

    print(f"\n===== {name1} vs {name2} FEATURE DISTRIBUTION SHIFT =====")
    print(out)

    return out


# ============================================================
# 8. EJECUCIÓN GENERAL
# ============================================================

def run_complete_extra_results(df, dataset_name, base_dir=DEFAULT_EXTRA_RESULTS_DIR):

    save_dir = base_dir
    make_dir(save_dir)

    print(f"\n========== EXTRA RESULTS {dataset_name} ==========")

    effect_size_pairwise(df, dataset_name, save_dir)
    auc_ranking(df, dataset_name, save_dir)
    model_error_analysis(df, dataset_name, save_dir)
    feature_importance(df, dataset_name, save_dir)
    minimal_feature_analysis(df, dataset_name, save_dir)
    histograms_by_class(df, dataset_name, save_dir)

    print(f"Resultados extra guardados en: {save_dir}")


def run_all_extra_results(df1, df2, base_dir=DEFAULT_EXTRA_RESULTS_DIR, name1="D1", name2="D2"):

    run_complete_extra_results(df1, name1, base_dir)
    run_complete_extra_results(df2, name2, base_dir)

    compare_datasets(df1, df2, base_dir, name1=name1, name2=name2)

    print("\n========== EXTRA ANALYSIS COMPLETO ==========")
    print(f"Resultados extra guardados en: {base_dir}")