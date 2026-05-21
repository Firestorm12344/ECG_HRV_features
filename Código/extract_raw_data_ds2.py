# ============================================================
# PROCESAR DATASET 2 - PTB-XL
# ============================================================

import os
import numpy as np
from Functions import (
    get_signals_ptbxl_dataset2,
    is_valid_signal_triplet,
    sep_data_stratified,
    save_dataset_splits,
    CLASS_NAMES
)

base_dir = r"C:\Users\juanz\Desktop\Universidad\ECG_data_bases"
save_dir = "Processed_Data"

dataset_path = os.path.join(
    base_dir,
    "D2 - ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

N = 5000
Num_signals = 2000
suffix = "2"

os.makedirs(save_dir, exist_ok=True)

signals, metadata, y = get_signals_ptbxl_dataset2(
    dataset_path=dataset_path,
    max_signals=None
)

print("Signals loaded:", len(signals))
print("Labels loaded:", len(y))

# ============================================================
# EXTRAER I, II, aVR
# En PTB-XL aVR aparece como AVR en mayúscula
# Índices: I=0, II=1, AVR=3
# ============================================================

I = []
II = []
aVR = []

for sig in signals:

    sig = np.asarray(sig, dtype=np.float32)

    lead_I = sig[:, 0]
    lead_II = sig[:, 1]
    lead_aVR = sig[:, 3]

    def normalize(x):
        max_abs = np.max(np.abs(x))
        if max_abs > 1e-8:
            return x / max_abs
        return x

    I.append(normalize(lead_I[:N]))
    II.append(normalize(lead_II[:N]))
    aVR.append(normalize(lead_aVR[:N]))

I = np.array(I, dtype=np.float32)
II = np.array(II, dtype=np.float32)
aVR = np.array(aVR, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("I:", I.shape)
print("II:", II.shape)
print("aVR:", aVR.shape)
print("y:", y.shape)

# ============================================================
# VALIDAR + BALANCEAR
# ============================================================

I_list = []
II_list = []
aVR_list = []
y_list = []

for cls in sorted(np.unique(y)):

    idx_cls = np.where(y == cls)[0]

    selected = []

    for idx in idx_cls:

        if is_valid_signal_triplet(I[idx], II[idx], aVR[idx]):
            selected.append(idx)

        if len(selected) >= Num_signals:
            break

    selected = np.array(selected, dtype=int)

    print(
        f"Clase {cls} ({CLASS_NAMES[int(cls)]}): "
        f"{len(selected)} seleccionadas"
    )

    I_list.append(I[selected])
    II_list.append(II[selected])
    aVR_list.append(aVR[selected])
    y_list.append(y[selected])

I_balanced = np.concatenate(I_list, axis=0)
II_balanced = np.concatenate(II_list, axis=0)
aVR_balanced = np.concatenate(aVR_list, axis=0)
y_balanced = np.concatenate(y_list, axis=0)

print("\nDistribución final:")
unique, counts = np.unique(y_balanced, return_counts=True)
print(dict(zip(unique, counts)))

# ============================================================
# GUARDAR XDATA / YDATA
# ============================================================

Xdata = np.stack([I_balanced, II_balanced, aVR_balanced], axis=2)
Ydata = y_balanced

np.save(os.path.join(save_dir, f"Xdata{suffix}.npy"), Xdata)
np.save(os.path.join(save_dir, f"Ydata{suffix}.npy"), Ydata)

print("Xdata:", Xdata.shape)
print("Ydata:", Ydata.shape)

# ============================================================
# SPLIT
# ============================================================

splits = sep_data_stratified(
    I=I_balanced,
    II=II_balanced,
    aVR=aVR_balanced,
    y=y_balanced,
    test_size=0.2,
    val_size=0.2,
    random_state=42
)

save_dataset_splits(
    data_dict=splits,
    save_dir=save_dir,
    suffix=suffix
)

print("\n---------- DATASET 2 READY -----------------")