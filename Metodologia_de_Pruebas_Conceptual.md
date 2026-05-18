# Metodología de Pruebas / Test Methodology

## 1. Objetivo / Objective

Este documento describe las pruebas estadísticas y análisis realizados para validar la significancia de las características HRV en la detección de arritmias cardíacas. Incluye análisis estadísticos, evaluación de modelos de machine learning y comparaciones entre datasets.

This document describes the statistical tests and analyses performed to validate the significance of HRV features in cardiac arrhythmia detection. It includes statistical analyses, machine learning model evaluation, and cross-dataset comparisons.

---

## 2. Preprocesamiento de Señales / Signal Preprocessing

- **Filtrado y Normalización**: Las señales ECG se filtran usando wavelet Daubechies 4 para eliminar ruido, seguido de normalización por el máximo absoluto.
- **Detección de Picos R**: Se identifican los picos R usando algoritmos de detección de picos con restricciones de distancia mínima (0.3 segundos) y prominencia.
- **Extracción de Características HRV**: Se calculan métricas de variabilidad del ritmo cardíaco:
  - Frecuencia cardíaca media (HR_mean)
  - Intervalo RR medio (RR_mean)
  - Desviación estándar de intervalos RR (SDNN)
  - Raíz cuadrada del promedio de diferencias cuadradas sucesivas de RR (RMSSD)

- **Filtering and Normalization**: ECG signals are filtered using Daubechies 4 wavelet to remove noise, followed by normalization by maximum absolute value.
- **R-peak Detection**: R-peaks are identified using peak detection algorithms with minimum distance constraints (0.3 seconds) and prominence.
- **HRV Feature Extraction**: Heart rate variability metrics are calculated:
  - Mean heart rate (HR_mean)
  - Mean RR interval (RR_mean)
  - Standard deviation of RR intervals (SDNN)
  - Root mean square of successive RR differences (RMSSD)
  - 

---

## 3. Análisis Estadísticos Descriptivos / Descriptive Statistical Analysis

- **Distribución de Clases**: Se analiza la distribución de muestras por clase (SR, AF, ST, SB) para verificar balance.
- **Estadísticas Descriptivas**: Se calculan medianas, medias, desviaciones estándar y rangos para cada característica HRV por clase.
- **Visualización**: Diagramas de caja (boxplots) para comparar distribuciones visualmente entre clases.

- **Class Distribution**: Sample distribution per class (SR, AF, ST, SB) is analyzed for balance.
- **Descriptive Statistics**: Medians, means, standard deviations, and ranges are calculated for each HRV feature per class.
- **Visualization**: Boxplots to visually compare distributions between classes.

---

## 4. Pruebas de Diferencias Estadísticas / Statistical Difference Tests

### 4.1 Prueba de Kruskal-Wallis

- **Propósito**: Evaluar si existen diferencias significativas en las distribuciones de cada característica HRV entre las cuatro clases de arritmia.
- **Método**: Prueba no paramétrica de Kruskal-Wallis para comparar múltiples grupos independientes.
- **Significancia**: Se considera significativo si p < 0.05.
- **Aplicación**: Se realiza por separado para cada característica HRV (HR_mean, RR_mean, SDNN, RMSSD).

- **Purpose**: Assess if there are significant differences in HRV feature distributions across the four arrhythmia classes.
- **Method**: Non-parametric Kruskal-Wallis test for comparing multiple independent groups.
- **Significance**: Considered significant if p < 0.05.
- **Application**: Performed separately for each HRV feature (HR_mean, RR_mean, SDNN, RMSSD).

### 4.2 Comparaciones Pareadas Post-hoc con Corrección Bonferroni

- **Propósito**: Identificar qué pares específicos de clases difieren significativamente en cada característica HRV.
- **Método**: Prueba de Mann-Whitney U para cada par de clases, con corrección de Bonferroni para controlar el error de tipo I en múltiples comparaciones.
- **Corrección**: El valor p se multiplica por el número total de pruebas (6 pares × 4 características = 24 pruebas).
- **Significancia**: Se considera significativo si p_bonferroni < 0.05.
- **Aplicación**: Se evalúan todos los pares de clases (SR-AF, SR-ST, SR-SB, AF-ST, AF-SB, ST-SB) para cada característica.

- **Purpose**: Identify which specific pairs of classes differ significantly in each HRV feature.
- **Method**: Mann-Whitney U test for each class pair, with Bonferroni correction to control type I error in multiple comparisons.
- **Correction**: p-value is multiplied by total number of tests (6 pairs × 4 features = 24 tests).
- **Significance**: Considered significant if p_bonferroni < 0.05.
- **Application**: All class pairs (SR-AF, SR-ST, SR-SB, AF-ST, AF-SB, ST-SB) are evaluated for each feature.

### 4.3 Tamaño del Efecto

- **Propósito**: Cuantificar la magnitud de las diferencias entre clases.
- **Método**: Estadístico d de Cohen para medir el tamaño del efecto en comparaciones pareadas.
- **Interpretación**: 
  - Pequeño: |d| < 0.2
  - Mediano: 0.2 ≤ |d| < 0.5
  - Grande: |d| ≥ 0.5
- **Aplicación**: Calculado para cada par de clases y característica HRV.

- **Purpose**: Quantify the magnitude of differences between classes.
- **Method**: Cohen's d statistic to measure effect size in pairwise comparisons.
- **Interpretation**:
  - Small: |d| < 0.2
  - Medium: 0.2 ≤ |d| < 0.5
  - Large: |d| ≥ 0.5
- **Application**: Calculated for each class pair and HRV feature.

---

## 5. Evaluación de Discriminabilidad / Discriminability Evaluation

### 5.1 Área Bajo la Curva ROC (AUC)

- **Propósito**: Evaluar la capacidad de cada característica HRV para discriminar entre clases (one-vs-rest).
- **Método**: Cálculo de AUC-ROC para cada característica y cada clase binaria.
- **Corrección de Dirección**: Si AUC < 0.5, se usa 1 - AUC para medir separabilidad independientemente de la dirección.
- **Interpretación**: AUC > 0.5 indica capacidad discriminativa; valores más altos indican mejor separación.
- **Aplicación**: Se calcula para cada característica HRV y cada clase.

- **Purpose**: Evaluate the ability of each HRV feature to discriminate between classes (one-vs-rest).
- **Method**: AUC-ROC calculation for each feature and each binary class.
- **Direction Correction**: If AUC < 0.5, 1 - AUC is used to measure separability regardless of direction.
- **Interpretation**: AUC > 0.5 indicates discriminative ability; higher values indicate better separation.
- **Application**: Calculated for each HRV feature and each class.

### 5.2 Ranking de Características

- **Propósito**: Ordenar las características HRV por su capacidad discriminativa promedio.
- **Método**: Promedio de AUC one-vs-rest para cada característica.
- **Aplicación**: Ayuda a identificar cuáles características son más informativas para la clasificación.

- **Purpose**: Rank HRV features by their average discriminative ability.
- **Method**: Average of one-vs-rest AUC for each feature.
- **Application**: Helps identify which features are most informative for classification.

---

## 6. Evaluación de Modelos de Machine Learning / Machine Learning Model Evaluation

### 6.1 Modelos Clásicos

- **Modelos Evaluados**: Regresión Logística y Random Forest.
- **Validación**: Validación cruzada estratificada de 5 folds para evitar sobreajuste.
- **Métricas**: Precisión (accuracy), F1-macro, desviaciones estándar.
- **Preprocesamiento**: Imputación de valores faltantes (mediana), escalado estándar.
- **Propósito**: Estimar el rendimiento máximo achievable con las características HRV.

- **Models Evaluated**: Logistic Regression and Random Forest.
- **Validation**: Stratified 5-fold cross-validation to prevent overfitting.
- **Metrics**: Accuracy, F1-macro, standard deviations.
- **Preprocessing**: Missing value imputation (median), standard scaling.
- **Purpose**: Estimate maximum achievable performance with HRV features.

### 6.2 Análisis de Características Mínimas

- **Propósito**: Determinar el número mínimo de características necesarias para mantener rendimiento.
- **Método**: Evaluación incremental agregando características en orden (RR_mean → HR_mean → SDNN → RMSSD).
- **Validación**: Validación cruzada con Regresión Logística.
- **Propósito**: Identificar redundancia en características y optimizar complejidad del modelo.

- **Purpose**: Determine minimum number of features needed to maintain performance.
- **Method**: Incremental evaluation adding features in order (RR_mean → HR_mean → SDNN → RMSSD).
- **Validation**: Cross-validation with Logistic Regression.
- **Purpose**: Identify feature redundancy and optimize model complexity.

### 6.3 Red Neuronal Profunda (DNN)

- **Arquitectura**: Red neuronal simple con capas densas, normalización por lotes y dropout.
- **Entrenamiento**: Optimizador Adam, función de pérdida categorical crossentropy, early stopping.
- **Validación**: Train-test split 80/20 estratificado.
- **Métricas**: Accuracy, F1-macro, reporte de clasificación, matriz de confusión.
- **Propósito**: Comparar rendimiento de deep learning con modelos clásicos usando solo características HRV.

- **Architecture**: Simple neural network with dense layers, batch normalization, and dropout.
- **Training**: Adam optimizer, categorical crossentropy loss, early stopping.
- **Validation**: Stratified 80/20 train-test split.
- **Metrics**: Accuracy, F1-macro, classification report, confusion matrix.
- **Purpose**: Compare deep learning performance with classical models using only HRV features.

---

## 7. Comparación entre Datasets / Cross-Dataset Comparison

### 7.1 Pruebas de Cambio de Distribución

- **Propósito**: Detectar diferencias sistemáticas entre datasets D1 y D2.
- **Métodos**:
  - Mann-Whitney U: Compara medianas entre datasets por clase y característica.
  - Kolmogorov-Smirnov: Evalúa diferencias en distribuciones completas.
- **Aplicación**: Se realiza por cada clase común y característica HRV.

- **Purpose**: Detect systematic differences between D1 and D2 datasets.
- **Methods**:
  - Mann-Whitney U: Compares medians between datasets per class and feature.
  - Kolmogorov-Smirnov: Assesses differences in complete distributions.
- **Application**: Performed for each common class and HRV feature.

### 7.2 Visualización Comparativa

- **Propósito**: Proporcionar evidencia visual de diferencias entre datasets.
- **Método**: Diagramas de caja comparativos D1 vs D2 por clase y característica.
- **Aplicación**: Facilita la identificación de sesgos o cambios en la distribución de datos.

- **Purpose**: Provide visual evidence of differences between datasets.
- **Method**: Comparative boxplots D1 vs D2 per class and feature.
- **Application**: Facilitates identification of data biases or distribution shifts.

---

## 8. Análisis Adicionales para Publicación / Additional Analysis for Publication

- **Importancia de Características**: Análisis de importancia usando Random Forest y coeficientes de Regresión Logística.
- **Análisis de Errores**: Matrices de confusión, reportes de clasificación detallados y tablas de errores por modelo.
- **Histogramas**: Distribuciones de frecuencia por clase para cada característica HRV.
- **Análisis Incremental**: Evaluación de rendimiento agregando características progresivamente con múltiples modelos.
- **Comparación Exhaustiva D1 vs D2**: Mann-Whitney, Cohen's d y estadísticas descriptivas entre datasets.

- **Feature Importance**: Importance analysis using Random Forest and Logistic Regression coefficients.
- **Error Analysis**: Confusion matrices, detailed classification reports, and error tables per model.
- **Histograms**: Frequency distributions per class for each HRV feature.
- **Incremental Analysis**: Performance evaluation adding features progressively with multiple models.
- **Comprehensive D1 vs D2 Comparison**: Mann-Whitney, Cohen's d, and descriptive statistics between datasets.

---

## 9. Consideraciones Generales / General Considerations

- **Corrección por Múltiples Comparaciones**: Se aplica corrección de Bonferroni en todas las pruebas post-hoc para controlar la tasa de falsos positivos.
- **Pruebas No Paramétricas**: Se usan Kruskal-Wallis y Mann-Whitney debido a que las distribuciones no siguen normalidad.
- **Validación Robusta**: Todos los modelos usan validación cruzada o splits estratificados para asegurar generalización.
- **Interpretabilidad**: Se prioriza la explicación de resultados sobre complejidad del modelo.

- **Multiple Comparison Correction**: Bonferroni correction is applied to all post-hoc tests to control false positive rate.
- **Non-parametric Tests**: Kruskal-Wallis and Mann-Whitney are used since distributions do not follow normality.
- **Robust Validation**: All models use cross-validation or stratified splits to ensure generalization.
- **Interpretability**: Result explanation is prioritized over model complexity.