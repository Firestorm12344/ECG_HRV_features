# ============================================================
# EXTRAER DATASET GENERAL USANDO TUS FUNCIONES
# Guarda:
# signals{dataset_id}.npy -> shape: (N, T, 3)
# labels{dataset_id}.npy  -> shape: (N,)
# ============================================================

import os
import numpy as np

from Functions import *

# ============================
# CONFIGURACIÓN
# ============================

dataset_id = 3

dataset_path = r"C:\Users\juanz\Desktop\Universidad\ECG_data_bases\D3 - classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
save_dir = os.path.join(ROOT_DIR, "Processed_Data")

N = 5000
num_Sig = 999999999

os.makedirs(save_dir, exist_ok=True)

# ============================
# LEER DATASET
# ============================

signals, all_meta = get_signals(
    path=dataset_path,
    num_Sig=num_Sig
)

print("Signals loaded:", len(signals))
print("Metadata loaded:", len(all_meta))

# ============================
# FILTRAR CLASES
# ============================

paired_signals, acronym_names, frequency_dict = get_diag(
    signals,
    all_meta,
    csv_file="Nombres.csv"
)

print("Frecuencia por clase [SR, AF, ST, SB]:")
print(frequency_dict)

# ============================
# EXTRAER I, II, aVR
# ============================

I, II, aVR, y = get_der(paired_signals)


# ============================
# RECORTAR / PAD A N MUESTRAS
# ============================

def crop_or_pad(sig, N):

    sig = np.asarray(sig, dtype=np.float32)

    if len(sig) >= N:
        return sig[:N]

    pad = np.zeros(N - len(sig), dtype=np.float32)

    return np.concatenate([sig, pad])

I = np.array(
    [crop_or_pad(x, N) for x in I],
    dtype=np.float32
)

II = np.array(
    [crop_or_pad(x, N) for x in II],
    dtype=np.float32
)

aVR = np.array(
    [crop_or_pad(x, N) for x in aVR],
    dtype=np.float32
)

y = np.array(y, dtype=np.int32)

print("\n===== ANTES DE VALIDAR =====")
unique, counts = np.unique(y, return_counts=True)
for cls, count in zip(unique, counts):
    print(CLASS_NAMES[int(cls)], ":", count)

valid_idx = []
invalid_count = {0: 0, 1: 0, 2: 0, 3: 0}
valid_count = {0: 0, 1: 0, 2: 0, 3: 0}

for i in range(len(y)):
    cls = int(y[i])

    if is_valid_signal_triplet(I[i], II[i], aVR[i]):
        valid_idx.append(i)
        valid_count[cls] += 1
    else:
        invalid_count[cls] += 1

valid_idx = np.array(valid_idx)

print("\n===== VÁLIDAS =====")
for cls in sorted(valid_count):
    print(CLASS_NAMES[cls], ":", valid_count[cls])

print("\n===== DESCARTADAS =====")
for cls in sorted(invalid_count):
    print(CLASS_NAMES[cls], ":", invalid_count[cls])



# ============================
# VALIDAR
# ============================
# ============================
# VALIDAR
# ============================

I = I[valid_idx]
II = II[valid_idx]
aVR = aVR[valid_idx]
y = y[valid_idx]

# ============================
# BALANCEAR A 1500 POR CLASE
# ============================

NUM_PER_CLASS = 1500

I_list = []
II_list = []
aVR_list = []
y_list = []

print("\n===== BALANCEO =====")

for cls in sorted(np.unique(y)):

    idx_cls = np.where(y == cls)[0]

    print(
        CLASS_NAMES[int(cls)],
        "disponibles:",
        len(idx_cls)
    )

    selected = idx_cls[:NUM_PER_CLASS]

    print(
        CLASS_NAMES[int(cls)],
        "seleccionadas:",
        len(selected)
    )

    I_list.append(I[selected])
    II_list.append(II[selected])
    aVR_list.append(aVR[selected])
    y_list.append(y[selected])

I = np.concatenate(I_list, axis=0)
II = np.concatenate(II_list, axis=0)
aVR = np.concatenate(aVR_list, axis=0)
y = np.concatenate(y_list, axis=0)

# ============================
# EMPAQUETAR
# ============================

signals_out = np.stack(
    [I, II, aVR],
    axis=2
)

labels_out = y

# ============================
# GUARDAR
# ============================

np.save(os.path.join(save_dir, f"signals{dataset_id}.npy"), signals_out)
np.save(os.path.join(save_dir, f"labels{dataset_id}.npy"), labels_out)

print("\nGuardado:")
print(f"signals{dataset_id}.npy:", signals_out.shape)
print(f"labels{dataset_id}.npy:", labels_out.shape)

CLASS_NAMES = {
    0: "SR",
    1: "AF",
    2: "ST",
    3: "SB"
}

unique, counts = np.unique(labels_out, return_counts=True)

print("\nDistribución final:")
for cls, count in zip(unique, counts):
    print(CLASS_NAMES[int(cls)], ":", count)