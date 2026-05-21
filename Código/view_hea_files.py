# ============================================================
# VER EJEMPLO DE .HEA POR DATASET
# ============================================================

import os

base_dir = r"C:\Users\juanz\Desktop\Universidad\ECG_data_bases"

dataset_folders = {
    1: "D1 -a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0",
    2: "D2 - ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    3: "D3 - classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2",
    4: "D4 - ptb-diagnostic-ecg-database-1.0.0"
}

for dataset_id, folder_name in dataset_folders.items():

    dataset_path = os.path.join(base_dir, folder_name)

    print("\n" + "=" * 80)
    print(f"DATASET {dataset_id}")
    print("=" * 80)

    found = False

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".hea"):

                hea_path = os.path.join(root, file)

                print("Archivo:", hea_path)
                print("-" * 80)

                with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                print(content)

                found = True
                break

        if found:
            break

    if not found:
        print("No se encontró ningún archivo .hea")