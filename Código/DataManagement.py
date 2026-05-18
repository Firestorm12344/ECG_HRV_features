import numpy as np
import os

from Functions import get_diag, get_der, sep_data_stratified, save_dataset_splits, is_valid_signal_triplet

# ============================  
# CONFIGURACIÓN
# ============================

data = 2  # Cambia a 2 para el segundo dataset

N = 10000
Num_signals = 2000
save_dir = "Processed_Data"

# ============================
# CARGA DE DATOS
# ============================

if data == 1:
    sig = np.load("Processed_Data/signals.npy")
    met = np.load("Processed_Data/metadata.npy", allow_pickle=True)
    suffix = "1"

elif data == 2:
    sig = np.load("Processed_Data/signals2.npy")
    met = np.load("Processed_Data/metadata2.npy", allow_pickle=True)
    suffix = "2"

else:
    raise ValueError("data debe ser 1 o 2")

print("Shape signals:", np.shape(sig))

# ============================
# OBTENER DIAGNÓSTICOS
# ============================

Pl, acro, freq = get_diag(sig, met)

print("Frecuencia por clase [SR, AF, ST, SB]:", freq)

# ============================
# OBTENER DERIVACIONES
# ============================

I, II, aVR, y = get_der(Pl)

I = np.array(I)[:, 0:N]
II = np.array(II)[:, 0:N]
aVR = np.array(aVR)[:, 0:N]
y = np.array(y)

print("I:", I.shape)
print("II:", II.shape)
print("aVR:", aVR.shape)
print("y:", y.shape)

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
        print(f"ADVERTENCIA: Clase {cls} solo tiene {len(selected)} señales válidas.")

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
        f"Clase {cls}: seleccionadas={rep['selected']}, "
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
print("ANTES DE SPLIT")
print("I_balanced:", I_balanced.shape)
print("II_balanced:", II_balanced.shape)
print("aVR_balanced:", aVR_balanced.shape)
print("y_balanced:", y_balanced.shape)

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