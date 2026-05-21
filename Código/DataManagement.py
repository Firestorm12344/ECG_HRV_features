import numpy as np
import os

from Functions import (
    sep_data_stratified,
    save_dataset_splits,
    is_valid_signal_triplet,
    CLASS_NAMES
)

# ============================
# CONFIGURACIÓN
# ============================

data = 3

N = 5000
Num_signals = 2000
save_dir = "Processed_Data"

# ============================
# CARGA DE DATOS
# ============================

sig = np.load(f"Processed_Data/signals{data}.npy")
y = np.load(f"Processed_Data/labels{data}.npy")

suffix = str(data)

print("Shape signals:", sig.shape)
print("Shape labels:", y.shape)

# ============================
# EXTRAER DERIVACIONES
# ============================

I = sig[:, 0:N, 0]
II = sig[:, 0:N, 1]
aVR = sig[:, 0:N, 2]

y = np.array(y, dtype=np.int32)

print("I:", I.shape)
print("II:", II.shape)
print("aVR:", aVR.shape)
print("y:", y.shape)

# ============================
# FRECUENCIA INICIAL
# ============================

unique, counts = np.unique(y, return_counts=True)

print("\nFrecuencia inicial:")
for cls, count in zip(unique, counts):
    print(CLASS_NAMES[int(cls)], ":", count)

# ============================
# BALANCEAR POR CLASE
# ============================

I_list = []
II_list = []
aVR_list = []
y_list = []

quality_report = {}

for cls in np.unique(y):

    idx_cls = np.where(y == cls)[0]

    selected = []
    discarded = 0

    for idx in idx_cls:

        if is_valid_signal_triplet(I[idx], II[idx], aVR[idx]):
            selected.append(idx)
        else:
            discarded += 1

        if len(selected) >= Num_signals:
            break

    selected = np.array(selected, dtype=int)

    quality_report[int(cls)] = {
        "selected": len(selected),
        "discarded_before_reaching_target": discarded,
        "target": Num_signals
    }

    if len(selected) < Num_signals:
        print(
            f"ADVERTENCIA: Clase {CLASS_NAMES[int(cls)]} "
            f"solo tiene {len(selected)} señales válidas."
        )

    I_list.append(I[selected])
    II_list.append(II[selected])
    aVR_list.append(aVR[selected])
    y_list.append(y[selected])

I_balanced = np.concatenate(I_list, axis=0)
II_balanced = np.concatenate(II_list, axis=0)
aVR_balanced = np.concatenate(aVR_list, axis=0)
y_balanced = np.concatenate(y_list, axis=0)

print("\n===== REPORTE DE CALIDAD =====")
for cls, rep in quality_report.items():
    print(
        f"{CLASS_NAMES[int(cls)]}: seleccionadas={rep['selected']}, "
        f"descartadas={rep['discarded_before_reaching_target']}, "
        f"objetivo={rep['target']}"
    )

print("\nDespués de balancear:")
print("I:", I_balanced.shape)
print("II:", II_balanced.shape)
print("aVR:", aVR_balanced.shape)
print("y:", y_balanced.shape)

unique, counts = np.unique(y_balanced, return_counts=True)
print("Distribución final:", dict(zip(unique, counts)))

# ============================
# SEPARACIÓN ESTRATIFICADA
# ============================

splits = sep_data_stratified(
    I=I_balanced,
    II=II_balanced,
    aVR=aVR_balanced,
    y=y_balanced,
    test_size=0.2,
    val_size=0.2,
    random_state=42
)

# ============================
# GUARDAR SPLITS
# ============================

save_dataset_splits(
    data_dict=splits,
    save_dir=save_dir,
    suffix=suffix
)

print("\n---------- DATA IS READY -----------------")

for key, value in splits.items():
    print(key + suffix, value.shape)