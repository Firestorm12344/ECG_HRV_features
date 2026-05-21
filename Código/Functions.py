
# ============================================================
# Functions.py
# Paper 1 - ECG HRV Features
# ============================================================
# Clases:
# 0 = SR
# 1 = AF
# 2 = ST
# 3 = SB
#
# Este archivo trabaja con la data YA guardada en Processed_Data:
# Ixtrain1.npy, IIxtrain1.npy, aVRxtrain1.npy, ytrain1.npy, etc.
#
# Las features HRV se extraen SOLO desde derivación II por defecto.
# ============================================================




import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.stats import kruskal, mannwhitneyu, ks_2samp
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wfdb

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


CLASS_NAMES = {
    0: "SR",
    1: "AF",
    2: "ST",
    3: "SB"
}




DEFAULT_FEATURES = ["RR_mean", "SDRR", "RMSSD", "NN50", "pNN50"]


# ============================================================
# NUEVA DETECCIÓN R + HRV
# REEMPLAZAR EN Functions.py
# ============================================================

def detect_r(ecg, fs=500, use_wavelet=True):

    import numpy as np

    # ========================================================
    # PREPROCESS
    # ========================================================

    x = preprocess_ecg_for_hrv(
        ecg,
        use_wavelet=use_wavelet,
        normalize=True
    )

    if len(x) == 0:
        return np.array([], dtype=int)

    if np.max(np.abs(x)) < 1e-8:
        return np.array([], dtype=int)

    # ========================================================
    # THRESHOLDS (HYSTERESIS)
    # ========================================================

    upper_threshold = 0.60
    lower_threshold = 0.45

    ignore_reading = False

    candidate_peaks = []

    # ========================================================
    # DETECTION
    # ========================================================

    for i in range(len(x)):

        value = x[i]

        if value >= upper_threshold and not ignore_reading:

            candidate_peaks.append(i)
            ignore_reading = True

        if value < lower_threshold:
            ignore_reading = False

    candidate_peaks = np.array(candidate_peaks)

    # ========================================================
    # REMOVE VERY CLOSE PEAKS
    # ========================================================

    min_distance = int(0.25 * fs)

    filtered_peaks = []

    last_peak = -999999

    for p in candidate_peaks:

        if p - last_peak >= min_distance:

            filtered_peaks.append(p)
            last_peak = p

    return np.array(filtered_peaks, dtype=int)


# ============================================================
# NUEVO EXTRACT_HRV
# ============================================================

def extract_hrv(ecg, fs=500, use_wavelet=True):

    import numpy as np

    peaks = detect_r(
        ecg,
        fs=fs,
        use_wavelet=use_wavelet
    )

    # ========================================================
    # VERY FEW PEAKS
    # ========================================================

    if len(peaks) < 2:
        return [np.nan, np.nan, np.nan, np.nan, np.nan]

    # ========================================================
    # RR INTERVALS
    # ========================================================

    rr = np.diff(peaks) / fs

    # ========================================================
    # PHYSIOLOGICAL FILTER
    # ========================================================

    rr = rr[
        (rr > 0.30) &
        (rr < 2.50)
    ]

    if len(rr) < 1:
        return [np.nan, np.nan, np.nan, np.nan, np.nan]

    # ========================================================
    # RR DIFFERENCES
    # ========================================================

    rr_diff = np.abs(np.diff(rr))

    # ========================================================
    # FEATURES
    # ========================================================

    RR_mean = np.mean(rr)

    SDRR = np.std(rr, ddof=1) if len(rr) > 1 else 0.0

    RMSSD = (
        np.sqrt(np.mean(rr_diff ** 2))
        if len(rr_diff) > 0
        else 0.0
    )

    NN50 = np.sum(rr_diff > 0.05)

    pNN50 = (
        (NN50 / len(rr_diff)) * 100
        if len(rr_diff) > 0
        else 0.0
    )

    return [
        RR_mean,
        SDRR,
        RMSSD,
        NN50,
        pNN50
    ]

# ------------------------------- DATA PROCESSING ------------------------------- #

def get_signals(path, num_Sig):
    signals = []
    all_meta = []  # To store metadata for each signal
    i = 0
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(path):
        for filename in files:
            if i<=num_Sig:
                if filename.endswith('.hea'):
                    base_filename = filename[:-4]
                    hea_filepath = os.path.join(root, base_filename)
                    
                    try:
                        
                        # Read the .hea file using wfdb.rdsamp
                        signal, meta = wfdb.rdsamp(hea_filepath)
                        signals.append(signal)
                        all_meta.append(meta)   
                        i+=1
                    except Exception as e:
                        print(f"Failed to read {hea_filepath}: {e}")
    return signals, all_meta



def getSingle_sigs(class_signal_number, signals, all_meta):
    class_signal = [class_signal_number]
    sig_list = []

    for i in range(len(all_meta)):
        diagnostics = all_meta[i]["comments"][2].split(" ")[1].split(",")
        for k in diagnostics:
            k_int = int(k)
            if k_int in class_signal:
                ind_signal = [signals[i]]
                sig_list.append(ind_signal)

    return sig_list
def get_diag(signals, all_meta, csv_file="Nombres.csv"):
    df = pd.read_csv(csv_file)
    acronym_names = df["Acronym Name"].tolist()

    pairedList = []

    AF = [164889003]
    ST = [427084000]
    SB = [426177001]
    SR = [426783006]

    frequency_dict = [0, 0, 0, 0]  # [SR, AF, ST, SB]

    for i in range(len(all_meta)):

        dx_comment = None

        for comment in all_meta[i]["comments"]:
            if comment.startswith("Dx:"):
                dx_comment = comment
                break

        if dx_comment is None:
            continue

        diagnostics = dx_comment.replace("Dx:", "").strip().split(",")

        for k in diagnostics:
            try:
                k_int = int(k.strip())
            except:
                continue

            if k_int in SR:
                code = 0
            elif k_int in AF:
                code = 1
            elif k_int in ST:
                code = 2
            elif k_int in SB:
                code = 3
            else:
                continue

            frequency_dict[code] += 1
            pairedList.append([signals[i], code])
            break

    return pairedList, acronym_names, frequency_dict


def get_der(PS):
    yDiagnosis = []
    ISignals = []
    IISignals = []
    aVRSignals = []

    for element in PS:
        signal = element[0]
        diagnostics = element[1]

        # Derivación I
        I = signal[:, 0]
        max_I = np.max(np.abs(I))
        if max_I > 0:
            I = I / max_I

        # Derivación II
        II = signal[:, 1]
        max_II = np.max(np.abs(II))
        if max_II > 0:
            II = II / max_II

        # aVR, índice 3 según tu código original
        aVR = signal[:, 3]
        max_aVR = np.max(np.abs(aVR))
        if max_aVR > 0:
            aVR = aVR / max_aVR

        ISignals.append(I)
        IISignals.append(II)
        aVRSignals.append(aVR)
        yDiagnosis.append(diagnostics)

    return ISignals, IISignals, aVRSignals, yDiagnosis


def is_valid_signal_triplet(I, II, aVR):
    signals = [I, II, aVR]

    for sig in signals:
        sig = np.asarray(sig)

        if np.any(np.isnan(sig)):
            return False

        if np.any(np.isinf(sig)):
            return False

        if len(sig) == 0:
            return False

        if np.max(np.abs(sig)) < 1e-8:
            return False

    return True


def sep_data_stratified(I, II, aVR, y, test_size=0.2, val_size=0.2, random_state=42):
    indices = np.arange(len(y))

    idx_trainval, idx_test = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    y_trainval = y[idx_trainval]

    val_relative_size = val_size / (1 - test_size)

    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_relative_size,
        stratify=y_trainval,
        random_state=random_state
    )

    data_dict = {
        "Ixtrain": I[idx_train],
        "IIxtrain": II[idx_train],
        "aVRxtrain": aVR[idx_train],
        "ytrain": y[idx_train],

        "Ixval": I[idx_val],
        "IIxval": II[idx_val],
        "aVRxval": aVR[idx_val],
        "yval": y[idx_val],

        "Ixtest": I[idx_test],
        "IIxtest": II[idx_test],
        "aVRxtest": aVR[idx_test],
        "ytest": y[idx_test],
    }

    return data_dict


def save_dataset_splits(data_dict, save_dir="Processed_Data", suffix="1"):
    os.makedirs(save_dir, exist_ok=True)

    for key, value in data_dict.items():
        save_path = os.path.join(save_dir, f"{key}{suffix}.npy")
        np.save(save_path, value)
        print("Guardado:", save_path, value.shape)

# ============================================================
# 1. LOAD DATA
# ============================================================

def load_split(processed_dir="Processed_Data", dataset=1, split="train"):
    """
    Carga un split previamente guardado por DataManagement.py.

    Parameters
    ----------
    processed_dir : str
        Carpeta donde están los .npy.
    dataset : int
        1 o 2.
    split : str
        train, val o test.

    Returns
    -------
    X : np.ndarray
        Señal multicanal con shape (N, T, 3), canales [I, II, aVR].
    y : np.ndarray
        Etiquetas.
    """

    suffix = str(dataset)

    I_path = os.path.join(processed_dir, f"Ix{split}{suffix}.npy")
    II_path = os.path.join(processed_dir, f"IIx{split}{suffix}.npy")
    aVR_path = os.path.join(processed_dir, f"aVRx{split}{suffix}.npy")
    y_path = os.path.join(processed_dir, f"y{split}{suffix}.npy")

    I = np.load(I_path).astype(np.float32)
    II = np.load(II_path).astype(np.float32)
    aVR = np.load(aVR_path).astype(np.float32)
    y = np.load(y_path).astype(int)

    # Asegurar shape (N, T)
    if I.ndim == 3:
        I = I[:, :, 0]
    if II.ndim == 3:
        II = II[:, :, 0]
    if aVR.ndim == 3:
        aVR = aVR[:, :, 0]

    X = np.stack([I, II, aVR], axis=2)  # (N, T, 3)

    return X, y


def load_all(processed_dir="Processed_Data", dataset=1):
    """
    Carga train + val + test juntos.
    """

    X_train, y_train = load_split(processed_dir, dataset, "train")
    X_val, y_val = load_split(processed_dir, dataset, "val")
    X_test, y_test = load_split(processed_dir, dataset, "test")

    X = np.concatenate([X_train, X_val, X_test], axis=0)
    y = np.concatenate([y_train, y_val, y_test], axis=0)

    return X, y


def load_dataset_all(processed_dir="Processed_Data", dataset=1):
    """
    Alias para compatibilidad con tu código.
    Hace lo mismo que load_all().
    """

    return load_all(processed_dir=processed_dir, dataset=dataset)


def check_dataset(X, y, name="Dataset"):
    """
    Revisión rápida de shape, NaN, Inf y distribución de clases.
    """

    print(f"\n===== {name} =====")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("NaN:", np.isnan(X).sum())
    print("Inf:", np.isinf(X).sum())
    print("min:", np.nanmin(X))
    print("max:", np.nanmax(X))
    print("mean:", np.nanmean(X))
    print("std:", np.nanstd(X))

    labels, counts = np.unique(y, return_counts=True)
    dist = {CLASS_NAMES.get(int(k), str(k)): int(v) for k, v in zip(labels, counts)}
    print("Distribución:", dist)


# ============================================================
# PREPROCESAMIENTO ECG
# ============================================================

def wavelet_denoise_db4(ecg, wavelet="db4", level=4, threshold=0.2):
    import numpy as np
    import pywt

    ecg = np.asarray(ecg, dtype=np.float32)
    ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)

    coeffs = pywt.wavedec(ecg, wavelet, level=level)

    filtered_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        filtered_coeffs.append(pywt.threshold(c, threshold, mode="soft"))

    filtered = pywt.waverec(filtered_coeffs, wavelet)

    return filtered[:len(ecg)]


def preprocess_ecg_for_hrv(ecg, use_wavelet=True, normalize=True):
    import numpy as np

    ecg = np.asarray(ecg, dtype=np.float32)
    ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)

    # 1) Quitar offset
    ecg = ecg - np.median(ecg)

    # 2) Filtrado wavelet db4
    if use_wavelet:
        ecg = wavelet_denoise_db4(ecg, wavelet="db4", level=4, threshold=0.2)

    # 3) Normalización por máximo absoluto
    if normalize:
        max_abs = np.max(np.abs(ecg))
        if max_abs > 1e-8:
            ecg = ecg / max_abs

    return ecg

def plot_r_peak_examples(
    X,
    y,
    fs=500,
    dataset_name="D1",
    lead_index=1,
    results_dir="Results/R_peaks_examples",
    samples_per_class=1
):
    """
    Grafica señales con picos R detectados.

    Por defecto usa lead_index=1, es decir, derivación II.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, T, 3).
    y : np.ndarray
        Etiquetas.
    fs : float
        Frecuencia de muestreo.
    dataset_name : str
        D1 o D2.
    lead_index : int
        0 = I, 1 = II, 2 = aVR.
    results_dir : str
        Carpeta de salida.
    samples_per_class : int
        Número de ejemplos por clase.
    """

    os.makedirs(results_dir, exist_ok=True)

    lead_names = ["I", "II", "aVR"]
    lead_name = lead_names[lead_index]

    for cls in sorted(np.unique(y)):
        idx_candidates = np.where(y == cls)[0]

        if len(idx_candidates) == 0:
            continue

        for n, idx in enumerate(idx_candidates[:samples_per_class]):
            ecg_raw = X[idx, :, lead_index]
            ecg = preprocess_ecg_for_hrv(ecg_raw, use_wavelet=True, normalize=True)
            peaks = detect_r(ecg_raw, fs=fs, use_wavelet=True)

            t = np.arange(len(ecg)) / fs

            plt.figure(figsize=(12, 4))
            plt.plot(t, ecg, linewidth=1, label=f"Lead {lead_name}")

            if len(peaks) > 0:
                plt.scatter(
                    t[peaks],
                    ecg[peaks],
                    color="red",
                    s=25,
                    label="Detected R peaks"
                )

            plt.title(
                f"{dataset_name} | Class: {CLASS_NAMES[int(cls)]} | "
                f"Lead: {lead_name} | R peaks: {len(peaks)}"
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Normalized amplitude")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()

            save_path = os.path.join(
                results_dir,
                f"{dataset_name}_{CLASS_NAMES[int(cls)]}_sample{n}_Rpeaks.png"
            )

            plt.savefig(save_path, dpi=300)
            plt.close()

            print("Guardado:", save_path)



def build_dataframe(X, y, fs=500, lead_index=1, dataset_name=None):
    """
    Construye DataFrame de features HRV.

    IMPORTANTE:
    Por defecto extrae features SOLO desde derivación II:
    lead_index = 1.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, T, 3), canales [I, II, aVR].
    y : np.ndarray
        Etiquetas.
    fs : float
        Frecuencia de muestreo.
    lead_index : int
        0 = I, 1 = II, 2 = aVR.
    dataset_name : str or None
        Nombre del dataset.

    Returns
    -------
    df : pd.DataFrame
    """

    data = []
    lead_names = ["I", "II", "aVR"]
    lead_name = lead_names[lead_index]

    for i in range(len(X)):
        ecg = X[i, :, lead_index]
        feats = extract_hrv(ecg, fs=fs)

        row = [
            i,
            int(y[i]),
            CLASS_NAMES[int(y[i])],
            lead_name,
            *feats
        ]

        if dataset_name is not None:
            row = [dataset_name] + row

        data.append(row)

    columns = ["id", "label", "class", "lead", "RR_mean", "SDRR", "RMSSD", "NN50", "pNN50"]

    if dataset_name is not None:
        columns = ["dataset"] + columns

    df = pd.DataFrame(data, columns=columns)

    return df


# ============================================================
# 4. STATS
# ============================================================

def run_stats(df, save_path=None, features=None):
    """
    Kruskal-Wallis por feature entre clases.
    """

    if features is None:
        features = DEFAULT_FEATURES

    rows = []

    for f in features:
        groups = []

        for c in sorted(df["label"].unique()):
            vals = df[df["label"] == c][f].dropna().values
            if len(vals) > 0:
                groups.append(vals)

        if len(groups) >= 2:
            H, p = kruskal(*groups)
        else:
            H, p = np.nan, np.nan

        rows.append({
            "feature": f,
            "H": H,
            "p": p,
            "significant_p_0_05": bool(p < 0.05) if np.isfinite(p) else False
        })

    out = pd.DataFrame(rows)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out.to_csv(save_path, index=False)

    return out


def run_pairwise_bonferroni(df, save_path=None, features=None):
    """
    Comparaciones par a par tipo post-hoc usando Mann-Whitney + Bonferroni.
    Es una alternativa práctica a Dunn test.
    """

    if features is None:
        features = DEFAULT_FEATURES

    rows = []
    classes = sorted(df["label"].unique())
    pairs = list(itertools.combinations(classes, 2))
    n_pairs = len(pairs)
    n_tests = n_pairs * len(features)

    for f in features:
        for c1, c2 in pairs:
            v1 = df[df["label"] == c1][f].dropna().values
            v2 = df[df["label"] == c2][f].dropna().values

            if len(v1) == 0 or len(v2) == 0:
                U, p = np.nan, np.nan
            else:
                U, p = mannwhitneyu(v1, v2, alternative="two-sided")

            p_bonf = min(p * n_tests, 1.0) if np.isfinite(p) else np.nan

            rows.append({
                "feature": f,
                "class_1": CLASS_NAMES[int(c1)],
                "class_2": CLASS_NAMES[int(c2)],
                "U": U,
                "p": p,
                "p_bonferroni": p_bonf,
                "significant_bonf_p_0_05": bool(p_bonf < 0.05) if np.isfinite(p_bonf) else False
            })

    out = pd.DataFrame(rows)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out.to_csv(save_path, index=False)

    return out


def dataset_shift_tests(df1, df2, features=None, results_dir=None):
    """
    Compara distribución D1 vs D2 por clase y feature.
    Usa Mann-Whitney y KS test.
    """

    if features is None:
        features = DEFAULT_FEATURES

    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
    rows = []

    common_classes = sorted(set(df1["label"].unique()).intersection(df2["label"].unique()))

    for f in features:
        for c in common_classes:
            v1 = df1[df1["label"] == c][f].dropna().values
            v2 = df2[df2["label"] == c][f].dropna().values

            if len(v1) == 0 or len(v2) == 0:
                u_p, ks_stat, ks_p = np.nan, np.nan, np.nan
            else:
                _, u_p = mannwhitneyu(v1, v2, alternative="two-sided")
                ks_stat, ks_p = ks_2samp(v1, v2)

            rows.append({
                "feature": f,
                "class": CLASS_NAMES[int(c)],
                "D1_median": np.nanmedian(v1) if len(v1) else np.nan,
                "D2_median": np.nanmedian(v2) if len(v2) else np.nan,
                "mannwhitney_p": u_p,
                "ks_statistic": ks_stat,
                "ks_p": ks_p
            })

    out = pd.DataFrame(rows)
    if results_dir is not None:
        out.to_csv(os.path.join(results_dir, "D1_vs_D2_feature_shift_tests.csv"), index=False)

    return out


# ============================================================
# 5. PLOTS
# ============================================================

def boxplots(df, save_dir, features=None):
    """
    Boxplots por feature y clase.
    """

    if features is None:
        features = DEFAULT_FEATURES

    os.makedirs(save_dir, exist_ok=True)

    for f in features:
        plt.figure(figsize=(7, 5))
        df.boxplot(column=f, by="class", showfliers=False)
        plt.title(f)
        plt.suptitle("")
        plt.xlabel("Class")
        plt.ylabel(f)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{f}.png"), dpi=300)
        plt.close()


def plot_D1_D2_boxplots(df_all, features=None, results_dir="Results/comparison/figures"):
    """
    Boxplots comparativos D1 vs D2 por clase.
    Requiere columna 'dataset'.
    """

    if features is None:
        features = DEFAULT_FEATURES

    os.makedirs(results_dir, exist_ok=True)

    if "dataset" not in df_all.columns:
        print("No existe columna dataset. No se pueden hacer boxplots D1 vs D2.")
        return

    for f in features:
        labels = []
        data = []

        for c in sorted(df_all["label"].unique()):
            for ds in sorted(df_all["dataset"].unique()):
                vals = df_all[(df_all["label"] == c) & (df_all["dataset"] == ds)][f].dropna().values
                labels.append(f"{CLASS_NAMES[int(c)]}-{ds}")
                data.append(vals)

        plt.figure(figsize=(10, 5))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.title(f"D1 vs D2 - {f}")
        plt.xlabel("Class-Dataset")
        plt.ylabel(f)
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"D1_D2_{f}.png"), dpi=300)
        plt.close()


# ============================================================
# 6. AUC
# ============================================================

def auc_features(df, features=None, output_path=None, direction_corrected=True):
    """
    Calcula AUC one-vs-rest por feature.

    direction_corrected=True:
    Si AUC < 0.5, usa 1 - AUC para medir separabilidad sin importar dirección.
    """

    if features is None:
        features = DEFAULT_FEATURES

    rows = []
    y = df["label"].values

    for f in features:
        x = df[f].values.astype(float)
        x = np.nan_to_num(x, nan=np.nanmedian(x), posinf=np.nanmedian(x), neginf=np.nanmedian(x))

        for c in sorted(np.unique(y)):
            yb = (y == c).astype(int)

            if len(np.unique(yb)) < 2:
                auc_val = np.nan
            else:
                auc_val = roc_auc_score(yb, x)

                if direction_corrected:
                    auc_val = max(auc_val, 1 - auc_val)

            row = {
                "feature": f,
                "class": CLASS_NAMES[int(c)],
                "auc": auc_val
            }
            rows.append(row)

            print(f"{f} vs {CLASS_NAMES[int(c)]} AUC={auc_val:.3f}")

    out = pd.DataFrame(rows)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out.to_csv(output_path, index=False)

    return out


# ============================================================
# 7. MODELS
# ============================================================

def train_models(df, features=None, output_path=None):
    """
    Modelos clásicos con validación cruzada:
    - Logistic Regression
    - Random Forest
    """

    if features is None:
        features = DEFAULT_FEATURES

    X = df[features].replace([np.inf, -np.inf], np.nan).values
    y = df["label"].values.astype(int)

    models = {
        "LR": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "RF": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for name, model in models.items():
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        row = {
            "model": name,
            "accuracy_mean": np.mean(acc_scores),
            "accuracy_std": np.std(acc_scores),
            "f1_macro_mean": np.mean(f1_scores),
            "f1_macro_std": np.std(f1_scores)
        }
        rows.append(row)

        print(name, "F1_macro:", np.mean(f1_scores), "+/-", np.std(f1_scores))

    out = pd.DataFrame(rows)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out.to_csv(output_path, index=False)

    return out


# ============================================================
# 8. MINIMAL FEATURES
# ============================================================

def minimal(df, features=None, output_path=None):
    """
    Evalúa desempeño agregando features progresivamente.
    """

    
    if features is None:
        features = DEFAULT_FEATURES

    rows = []

    for k in range(1, len(features) + 1):
        feats = features[:k]

        X = df[feats].replace([np.inf, -np.inf], np.nan).values
        y = df["label"].values.astype(int)

        model = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ])

        scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")

        row = {
            "n_features": k,
            "features_used": ", ".join(feats),
            "f1_macro_mean": np.mean(scores),
            "f1_macro_std": np.std(scores)
        }

        rows.append(row)
        print(f"{k} features:", np.mean(scores))

    out = pd.DataFrame(rows)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out.to_csv(output_path, index=False)

    return out


# ============================================================
# 9. DNN SIMPLE CON FEATURES
# ============================================================
def plot_combined_boxplots(df1, df2, features, results_dir):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(results_dir, exist_ok=True)

    df1 = df1.copy()
    df2 = df2.copy()

    df1 = df1.loc[:, ~df1.columns.duplicated()]
    df2 = df2.loc[:, ~df2.columns.duplicated()]

    if "dataset" in df1.columns:
        df1 = df1.drop(columns=["dataset"])
    if "dataset" in df2.columns:
        df2 = df2.drop(columns=["dataset"])

    df1["dataset"] = "D1"
    df2["dataset"] = "D2"

    df_all = pd.concat([df1, df2], axis=0, ignore_index=True)

    class_order = ["SR", "AF", "ST", "SB"]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12, 10)
    )

    axes = axes.flatten()

    for i, feature in enumerate(features):

        plot_df = df_all[["class", "dataset", feature]].copy()
        plot_df = plot_df.replace([float("inf"), -float("inf")], pd.NA)
        plot_df = plot_df.dropna(subset=[feature])

        sns.boxplot(
            data=plot_df,
            x="class",
            y=feature,
            hue="dataset",
            order=class_order,
            ax=axes[i],
            showfliers=False
        )

        axes[i].set_title(feature)
        axes[i].set_xlabel("Rhythm class")
        axes[i].set_ylabel(feature)
        axes[i].grid(alpha=0.3)

        if i != 0:
            axes[i].get_legend().remove()
        else:
            axes[i].legend(title="Dataset", loc="best")

    # Apagar subplot vacío
    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    save_path = os.path.join(results_dir, "combined_boxplots_D1_D2_grid.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Boxplot combinado guardado en: {save_path}")

def evaluate_dnn_feature_model(df, features=None, results_dir=None, prefix="D1"):
    """
    DNN simple usando únicamente features HRV.
    Sirve para evaluar si el bajo rendimiento se debe al modelo
    o a la capacidad informativa limitada de las features.
    """

    if features is None:
        features = DEFAULT_FEATURES

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam

    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)

    X = df[features].replace([np.inf, -np.inf], np.nan).values
    y = df["label"].values.astype(int)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    y_train_cat = to_categorical(y_train, num_classes=4)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(16, activation="relu"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    early = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train_cat,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=[early],
        verbose=0
    )

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    metrics = {
        "dataset": prefix,
        "model": "Simple_DNN_features",
        "n_features": len(features),
        "accuracy": acc,
        "f1_macro": f1_macro
    }

    df_metrics = pd.DataFrame([metrics])
    if results_dir is not None:
        df_metrics.to_csv(os.path.join(results_dir, f"{prefix}_DNN_feature_model_results.csv"), index=False)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["SR", "AF", "ST", "SB"],
        output_dict=True,
        zero_division=0
    )

    if results_dir is not None:
        pd.DataFrame(report).transpose().to_csv(
            os.path.join(results_dir, f"{prefix}_DNN_classification_report.csv")
        )

        cm = confusion_matrix(y_test, y_pred)
        pd.DataFrame(
            cm,
            index=["SR", "AF", "ST", "SB"],
            columns=["SR", "AF", "ST", "SB"]
        ).to_csv(
            os.path.join(results_dir, f"{prefix}_DNN_confusion_matrix.csv")
        )

    print(f"\n===== DNN FEATURE MODEL {prefix} =====")
    print(df_metrics)
    print(classification_report(y_test, y_pred, target_names=["SR", "AF", "ST", "SB"], zero_division=0))

    return metrics


def analyze_dataset(dataset_id, dataset_name):
    print(f"\n==================== {dataset_name} ====================")

    X, y = load_dataset_all(
        processed_dir=processed_dir,
        dataset=dataset_id
    )

    check_dataset(X, y, name=dataset_name)

    # ----------------------------
    # Validación visual picos R
    # ----------------------------
    plot_r_peak_examples(
        X=X,
        y=y,
        fs=fs,
        dataset_name=dataset_name,
        lead_index=lead_index,
        results_dir=os.path.join(results_dir, "R_peaks_examples"),
        samples_per_class=3
    )

    # ----------------------------
    # Extraer HRV
    # ----------------------------
    df = build_dataframe(
        X=X,
        y=y,
        fs=fs,
        lead_index=lead_index,
        dataset_name=dataset_name
    )

    df.to_csv(f"Results/{dataset_name}/tables/features.csv", index=False)
    pd.Series(y).value_counts().to_csv(
        f"Results/{dataset_name}/tables/class_distribution.csv"
    )

    # ----------------------------
    # Estadística
    # ----------------------------
    print("\n=== STATS D"+str(dataset_id)+" ===")

    stats_d = run_stats(
        df=df,
        save_path="Results/D"+str(dataset_id)+"/tables/kruskal.csv",
        features=features
    )

    print(stats_d)

    bonf_d = run_pairwise_bonferroni(
        df=df,
        save_path="Results/D"+str(dataset_id)+"/tables/bonferroni.csv",
        features=features
    )

    print(bonf_d.head())

    # ----------------------------
    # AUC por feature
    # ----------------------------
    auc_features(
        df=df,
        features=features,
        output_path=f"Results/{dataset_name}/tables/auc_features.csv"
    )

    # ----------------------------
    # Modelos clásicos
    # ----------------------------
    train_models(
        df=df,
        features=features,
        output_path=f"Results/{dataset_name}/tables/model_results.csv"
    )

    # ----------------------------
    # Features mínimas
    # ----------------------------
    minimal(
        df=df,
        features=features,
        output_path=f"Results/{dataset_name}/tables/minimal_features.csv"
    )

    # ----------------------------
    # Boxplots
    # ----------------------------
    boxplots(
        df=df,
        save_dir=f"Results/{dataset_name}/figures",
        features=features
    )

    # ----------------------------
    # DNN con features
    # ----------------------------
    evaluate_dnn_feature_model(
        df=df,
        features=features,
        results_dir=f"Results/{dataset_name}/tables",
        prefix=dataset_name
    )

    return X, y, df