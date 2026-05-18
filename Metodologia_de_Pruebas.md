# Metodología de Pruebas / Test Methodology

## 1. Objetivo / Objective

Este documento describe la metodología de prueba utilizada en el proyecto de clasificación de arritmias basado en señales ECG y características HRV. Incluye los pasos de preprocesamiento, extracción de características, pruebas estadísticas, evaluación de modelos clásicos y análisis adicionales de comparación entre datasets.

This document describes the test methodology used in the ECG arrhythmia classification project. It includes preprocessing, feature extraction, statistical tests, classical model evaluation, and additional cross-dataset analysis.

---

## 2. Datos y carga / Data and Loading

- Se utilizan datasets preprocesados que ya están guardados en la carpeta `Processed_Data`.
- Cada dataset incluye señales ECG de tres derivaciones: I, II y aVR.
- Las señales se cargan como matrices `X` con forma `(N, T, 3)` y etiquetas `y`.
- Los datasets D1 y D2 se procesan por separado y luego se concatenan para comparación.

- Preprocessed datasets are stored in `Processed_Data`.
- Each dataset contains ECG signals from three leads: I, II, and aVR.
- Signals are loaded as arrays `X` with shape `(N, T, 3)` and labels `y`.
- Datasets D1 and D2 are analyzed separately and later concatenated for comparison.

---

## 3. Preprocesamiento de ECG / ECG Preprocessing

- Se utiliza la derivación II de forma predeterminada para la extracción de características HRV.
- La señal ECG se normaliza y se filtra con un denoising por wavelet Daubechies 4 (`db4`).
- El proceso incluye:
  - eliminación de offset (restar la mediana)
  - denoising wavelet con `pywt`
  - normalización por el máximo absoluto

- Lead II is used by default for HRV feature extraction.
- ECG signals are normalized and denoised using Daubechies 4 (`db4`) wavelet.
- Preprocessing steps:
  - remove signal offset (subtract median)
  - wavelet denoising using `pywt`
  - normalize by maximum absolute value

---

## 4. Detección de picos R / R-peak Detection

- Se detectan picos R en la señal procesada.
- Se usa `scipy.signal.find_peaks` con:
  - distancia mínima entre picos (`min_distance_sec = 0.30`)
  - prominencia mínima
- Se comparan picos positivos y negativos, y se selecciona el conjunto con mayor prominencia total.

- R-peaks are detected in the processed ECG signal.
- `scipy.signal.find_peaks` is used with:
  - minimum distance between peaks (`min_distance_sec = 0.30`)
  - minimum prominence
- Positive and negative peaks are compared, and the set with the greater total prominence is selected.

---

## 5. Extracción de características HRV / HRV Feature Extraction

- Se construye un `DataFrame` con las características HRV extraídas por señal.
- Características calculadas:
  - `HR_mean` (frecuencia cardíaca media)
  - `RR_mean` (intervalo RR medio)
  - `SDNN` (desviación estándar de los intervalos RR)
  - `RMSSD` (raíz cuadrada del promedio de las diferencias al cuadrado entre intervalos RR consecutivos)
- Cada registro incluye la etiqueta de clase y la derivación usada.

- A `DataFrame` is built with HRV features extracted from each signal.
- Features calculated:
  - `HR_mean` (mean heart rate)
  - `RR_mean` (mean RR interval)
  - `SDNN` (standard deviation of RR intervals)
  - `RMSSD` (root mean square of successive RR differences)
- Each record includes label and lead information.

---

## 6. Pruebas estadísticas / Statistical Tests

### 6.1 Kruskal-Wallis

- Se ejecuta `run_stats()` para cada dataset.
- Prueba `Kruskal-Wallis` por cada característica HRV entre todas las clases.
- Objetivo: verificar si existen diferencias significativas en la distribución de la característica entre clases.
- Resultados guardados en `Results/D1/tables/stats.csv` y `Results/D2/tables/stats.csv`.

### 6.1 Kruskal-Wallis

- `run_stats()` is executed for each dataset.
- The Kruskal-Wallis test is applied to each HRV feature across all classes.
- Objective: check if feature distributions differ significantly between classes.
- Results are saved in `Results/D1/tables/stats.csv` and `Results/D2/tables/stats.csv`.

### 6.2 Comparaciones pareadas con Bonferroni / Pairwise Bonferroni Comparisons

- Se ejecuta `run_pairwise_bonferroni()` para cada dataset.
- Se usa la prueba de `Mann-Whitney U` entre cada par de clases.
- Corrección de Bonferroni aplicada al valor `p`.
- Objetivo: identificar pares de clases con diferencias significativas en cada característica.
- Resultados guardados en `Results/D1/tables/pairwise_bonferroni.csv` y `Results/D2/tables/pairwise_bonferroni.csv`.

### 6.2 Pairwise Bonferroni Comparisons

- `run_pairwise_bonferroni()` is executed for each dataset.
- Mann-Whitney U test is used for each pair of classes.
- Bonferroni correction is applied to p-values.
- Objective: identify class pairs with significant feature differences.
- Results are saved in `Results/D1/tables/pairwise_bonferroni.csv` and `Results/D2/tables/pairwise_bonferroni.csv`.

---

## 7. Evaluación de discriminabilidad / Discriminability Evaluation

### 7.1 AUC por característica / Feature AUC

- Se calcula `auc_features()` para cada dataset.
- Para cada característica y cada clase, se evalúa AUC `one-vs-rest`.
- Si AUC < 0.5, se corrige como `1 - AUC` para reflejar separabilidad independiente de la dirección.
- Objetivo: medir la capacidad discriminativa individual de cada métrica HRV.
- Resultados guardados en `Results/D1/tables/auc_features.csv` y `Results/D2/tables/auc_features.csv`.

### 7.1 Feature AUC

- `auc_features()` is computed for each dataset.
- For each feature and each class, one-vs-rest AUC is evaluated.
- If AUC < 0.5, it is corrected to `1 - AUC` to measure separability regardless of direction.
- Objective: measure the discriminative power of each HRV metric.
- Results are saved in `Results/D1/tables/auc_features.csv` and `Results/D2/tables/auc_features.csv`.

---

## 8. Evaluación de modelos clásicos / Classical Model Evaluation

### 8.1 Modelos entrenados / Trained Models

- Se usa `train_models()` con las características HRV.
- Modelos evaluados:
  - Logistic Regression
  - Random Forest
- Se aplica validación cruzada estratificada de 5 folds.
- Métricas calculadas:
  - `accuracy_mean`, `accuracy_std`
  - `f1_macro_mean`, `f1_macro_std`
- Objetivo: estimar la capacidad de clasificación de las features HRV.
- Resultados guardados en `Results/D1/tables/model_results.csv` y `Results/D2/tables/model_results.csv`.

### 8.1 Trained Models

- `train_models()` is used with HRV features.
- Models evaluated:
  - Logistic Regression
  - Random Forest
- Stratified 5-fold cross-validation is applied.
- Metrics computed:
  - `accuracy_mean`, `accuracy_std`
  - `f1_macro_mean`, `f1_macro_std`
- Objective: estimate classification performance from HRV features.
- Results are saved in `Results/D1/tables/model_results.csv` and `Results/D2/tables/model_results.csv`.

### 8.2 Análisis de características mínimas / Minimal Feature Analysis

- Se ejecuta `minimal()` para evaluar rendimiento acumulando features.
- Se agregan características en el orden: `RR_mean`, `HR_mean`, `SDNN`, `RMSSD`.
- Para cada subconjunto `k = 1..4`, se entrena un Logistic Regression y se evalúa con `f1_macro`.
- Objetivo: determinar cuántas y cuáles features son suficientes para mantener rendimiento.
- Resultados guardados en `Results/D1/tables/minimal_features.csv` y `Results/D2/tables/minimal_features.csv`.

### 8.2 Minimal Feature Analysis

- `minimal()` is run to evaluate performance with incremental feature sets.
- Features are added in order: `RR_mean`, `HR_mean`, `SDNN`, `RMSSD`.
- For each subset `k = 1..4`, a Logistic Regression model is trained and evaluated with `f1_macro`.
- Objective: determine the minimum number of features needed for good performance.
- Results are saved in `Results/D1/tables/minimal_features.csv` and `Results/D2/tables/minimal_features.csv`.

---

## 9. Evaluación de DNN / DNN Evaluation

- Se evalúa `evaluate_dnn_feature_model()` con las mismas features HRV.
- Arquitectura utilizada:
  - Entrada compatible con el número de features
  - Dense(32) + BatchNormalization + Dropout(0.25)
  - Dense(16) + BatchNormalization + Dropout(0.25)
  - Capa final softmax con 4 salidas
- Se usa `train_test_split` 80/20 con estratificación y `EarlyStopping`.
- Métricas guardadas:
  - `accuracy`
  - `f1_macro`
  - Classification report
  - Confusion matrix
- Objetivo: comparar rendimiento de un modelo de deep learning simple con los modelos clásicos.

- `evaluate_dnn_feature_model()` is evaluated with the same HRV features.
- Architecture:
  - Input layer matching the number of features
  - Dense(32) + BatchNormalization + Dropout(0.25)
  - Dense(16) + BatchNormalization + Dropout(0.25)
  - Softmax output layer with 4 classes
- `train_test_split` 80/20 is used with stratification and `EarlyStopping`.
- Metrics saved:
  - `accuracy`
  - `f1_macro`
  - classification report
  - confusion matrix
- Objective: compare a simple deep learning model against classical models.

---

## 10. Comparación entre datasets D1 y D2 / Dataset Comparison D1 vs D2

### 10.1 Prueba de cambio de distribución / Shift Tests

- Se ejecuta `dataset_shift_tests()`.
- Compara D1 vs D2 por cada feature y por cada clase común.
- Pruebas usadas:
  - Mann-Whitney U para diferencias de ubicación
  - Kolmogorov-Smirnov para diferencias de distribución
- Objetivo: detectar sesgos, cambios de distribución o diferencias sistemáticas entre D1 y D2.
- Resultados guardados en `Results/comparison/tables/D1_vs_D2_feature_shift_tests.csv`.

### 10.1 Shift Tests

- `dataset_shift_tests()` is executed.
- Compares D1 vs D2 for each feature and each common class.
- Tests used:
  - Mann-Whitney U for location differences
  - Kolmogorov-Smirnov for distribution differences
- Objective: detect dataset bias, distribution changes, or systematic differences between D1 and D2.
- Results are saved in `Results/comparison/tables/D1_vs_D2_feature_shift_tests.csv`.

### 10.2 Boxplots comparativos / Comparative Boxplots

- Se generan boxplots con `plot_D1_D2_boxplots()`.
- Se visualiza la distribución de cada feature por clase y por dataset.
- Objetivo: ofrecer evidencia visual de diferencias entre D1 y D2.
- Figuras guardadas en `Results/comparison/figures`.

### 10.2 Comparative Boxplots

- `plot_D1_D2_boxplots()` generates comparative boxplots.
- Visualizes feature distributions by class and dataset.
- Objective: provide visual evidence of differences between D1 and D2.
- Figures are saved in `Results/comparison/figures`.

---

## 11. Análisis extra para paper / Extra Analysis for Paper

- `run_all_extra_results()` ejecuta análisis adicionales para ambos datasets y la comparación.
- Incluye:
  - `effect_size_pairwise`: tamaño de efecto y pruebas Mann-Whitney con corrección Bonferroni
  - `auc_ranking`: ranking de features según AUC mean one-vs-rest
  - `model_error_analysis`: métricas, informes de clasificación, matrices de confusión y análisis de errores para Logistic Regression y Random Forest
  - `feature_importance`: importancia de características con Random Forest y coeficientes de Logistic Regression
  - `minimal_feature_analysis`: análisis incremental de features con Logistic Regression y Random Forest
  - `histograms_by_class`: histogramas de distribución por clase para cada feature
  - `compare_datasets`: comparación adicional D1 vs D2 con Mann-Whitney y Cohen's d
- Objetivo: generar resultados exhaustivos para soporte de paper y comparación entre datasets.

- `run_all_extra_results()` runs additional analysis for both datasets and comparison.
- Includes:
  - `effect_size_pairwise`: effect size and Mann-Whitney tests with Bonferroni correction
  - `auc_ranking`: feature ranking by mean one-vs-rest AUC
  - `model_error_analysis`: metrics, classification reports, confusion matrices, and error analysis for Logistic Regression and Random Forest
  - `feature_importance`: feature importance from Random Forest and coefficients from Logistic Regression
  - `minimal_feature_analysis`: incremental feature analysis with Logistic Regression and Random Forest
  - `histograms_by_class`: histogram distributions by class for each feature
  - `compare_datasets`: additional D1 vs D2 comparison using Mann-Whitney and Cohen's d
- Objective: produce exhaustive results to support the paper and dataset comparison.

---

## 12. Salidas principales / Main Outputs

- `Results/D1/tables/`
  - `features.csv`
  - `stats.csv`
  - `pairwise_bonferroni.csv`
  - `auc_features.csv`
  - `model_results.csv`
  - `minimal_features.csv`
  - `D1_DNN_*` (informes DNN)
- `Results/D2/tables/`
  - mismos archivos para D2
- `Results/comparison/tables/`
  - `features_D1_D2.csv`
  - `D1_vs_D2_feature_shift_tests.csv`
- `Results/comparison/figures/`
  - boxplots comparativos D1 vs D2
- `Results/extra_results/`
  - análisis extendido para la publicación

- `Results/D1/tables/`
  - `features.csv`
  - `stats.csv`
  - `pairwise_bonferroni.csv`
  - `auc_features.csv`
  - `model_results.csv`
  - `minimal_features.csv`
  - `D1_DNN_*` (DNN reports)
- `Results/D2/tables/`
  - same files for D2
- `Results/comparison/tables/`
  - `features_D1_D2.csv`
  - `D1_vs_D2_feature_shift_tests.csv`
- `Results/comparison/figures/`
  - comparative D1 vs D2 boxplots
- `Results/extra_results/`
  - extended analysis for the paper
