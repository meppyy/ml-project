# Kerala Flood-Year Classification Using Machine Learning

## 1. Project Overview

This project investigates whether historical monthly rainfall data can be used to classify documented major flood years in Kerala using classical machine-learning algorithms.

The study examines how classification performance changes when rainfall information is restricted to different prediction cutoffs throughout the year.

**Domain:** Disaster Management  
**SDGs:** SDG 11 – Sustainable Cities and Communities; SDG 13 – Climate Action  
**ML Task:** Binary Classification  
**Algorithms:** Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM  
**Dataset Period:** 1901–2017  
**Article Type:** Short Communication

## 2. Problem Statement

The 2018 Kerala floods caused extensive loss of life and displacement. This project explores whether historical monthly rainfall observations can help distinguish documented major flood years from other years.

The rainfall dataset covers 1901–2017. Therefore, 2018 is used only as contextual motivation and is not included as a model observation.

The study is retrospective and exploratory. It does not claim to provide an operational flood-warning system.

## 3. Dataset

**Dataset:** Rainfall in Kerala (1901–2017)

**Source:** Kaggle, based on India Meteorological Department (IMD) rainfall data.

Dataset:
https://www.kaggle.com/datasets/ggopinathan/rainfall-in-kerala-from-19012017

The dataset contains:
- 117 annual observations.
- 19 original columns.
- Monthly rainfall measurements.
- Annual rainfall totals.
- Seasonal rainfall aggregates.

The dataset represents the Kerala subdivision.

## 4. Flood-Year Labelling

The binary target variable is `FLOOD_LABEL`.

- `1`: Year documented as a major flood year in the selected Kerala State Planning Board chronology.
- `0`: Year not listed in the selected chronology.

Documented positive years:

1924, 1958, 1961, 1992, 2007 and 2013.

### Class Distribution

| Class | Observations | Percentage |
|---|---:|---:|
| Documented major flood | 6 | 5.13% |
| Not listed | 111 | 94.87% |
| Total | 117 | 100% |

A zero label does not establish that no flooding occurred during that year. It only indicates that the year was not included in the selected major-flood chronology.

The labelled dataset is stored in:

`data/processed/kerala_flood_labeled.csv`

## 5. Exploratory Data Analysis

The EDA examines:

- Dataset structure and data quality.
- Missing values and duplicate records.
- Annual rainfall trends.
- Average monthly rainfall.
- Seasonal rainfall distribution.
- Monthly rainfall variability.
- Correlations between monthly rainfall variables.
- Rainfall differences between documented flood years and other years.

The analysis identifies substantial seasonal variation in Kerala rainfall, with June–September contributing the largest share of annual rainfall.

Four of the six documented flood years rank above the 90th percentile of JJAS rainfall.

## 6. Machine-Learning Methods

Five classical classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbours (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Preprocessing Pipeline

The classification pipeline consists of:

1. StandardScaler
2. RandomOverSampler
3. Classification algorithm

Scaling and random oversampling are applied within the training folds to prevent validation data from influencing preprocessing.

## 7. Prediction Cutoffs

Five prediction windows were investigated.

| Cutoff | Available rainfall features |
|---|---|
| March | January–March |
| May | January–May |
| June | January–June |
| July | January–July |
| September | January–September |

Only rainfall observations available up to each selected cutoff are used as model features.

Annual and seasonal aggregates containing information from later months are excluded from these prediction experiments.

## 8. Model Evaluation

Models were evaluated using stratified three-fold cross-validation.

**Configuration:**
- Number of folds: 3
- Shuffle: True
- Random state: 42

**Evaluation metrics:**
- Recall
- Precision
- F1-score
- ROC-AUC

Pooled out-of-fold predictions are used for the main model comparison.

### Best Observed Results

| Metric | Model and cutoff | Score |
|---|---|---:|
| Recall | Logistic Regression – June | 0.667 |
| F1-score | Logistic Regression – June | 0.296 |
| ROC-AUC | Logistic Regression – September | 0.737 |

Logistic Regression using January–June rainfall achieved the best recall and F1-score among the tested combinations.

These results are exploratory cross-validation estimates and are not independent test-set results.

### Additional Evaluation

The project also includes:

- Individual rainfall-feature ROC-AUC comparisons.
- Fold-level confusion counts and evaluation metrics.
- Fold-level mean and standard deviation summaries.
- A 95% Wilson confidence interval for the June Logistic Regression recall.

The rainfall-only baseline analysis uses all 117 observations and is descriptive, not an independent predictive evaluation.

## 9. Project Structure

```text
ml-project/
│
├── data/
│   ├── raw/
│   │   ├── Kerala-Rainfall-Historical.csv
│   │   └── flood_year_evidence.csv
│   │
│   └── processed/
│       ├── kerala_flood_labeled.csv
│       ├── model_comparison_results.csv
│       ├── oof_predictions.csv
│       ├── fold_level_metrics.csv
│       ├── fold_variability_summary.csv
│       └── rainfall_only_baseline_auc.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_flood_year_labeling.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── reports/
│   └── figures/
│       ├── annual_rainfall_trend.png
│       ├── monthly_rainfall.png
│       ├── seasonal_rainfall.png
│       ├── monthly_boxplot.png
│       ├── rainfall_correlation.png
│       ├── flood_vs_nonflood_rainfall.png
│       ├── f1_comparison.png
│       ├── logistic_cutoff_metrics.png
│       ├── june_roc_curves.png
│       └── june_confusion_matrix.png
│
├── requirements.txt
└── .gitignore
```

## 10. Installation and Execution

### Clone the repository

```bash
git clone https://github.com/meppyy/ml-project.git
cd ml-project
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate on Windows

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Execute the notebooks in numerical order:

1. `01_data_exploration.ipynb`
2. `02_flood_year_labeling.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_model_training.ipynb`

The notebooks generate the processed datasets, evaluation results and visualisations stored in the repository.

## 11. Limitations

- The dataset contains only 117 annual observations.
- Only six years are labelled as documented major flood years.
- The dataset has a highly imbalanced class distribution.
- Shuffled stratified cross-validation does not simulate forecasting future years.
- Fold-level metrics are sensitive to individual flood-year observations.
- State-level monthly rainfall aggregates cannot capture all local hydrological and meteorological factors.
- The study does not establish operational flood-warning capability.

Results should therefore be interpreted as exploratory historical classification findings.

## 12. References

1. Government of Kerala, Kerala State Planning Board, *Final Report of the Committee to Examine the Causes of Repeated Extreme Heavy Rainfall Events, Subsequent Floods and Landslides*, December 2019. See Table 5.7.

2. G. Gopinathan, *Rainfall in Kerala from 1901–2017*, Kaggle dataset, based on IMD rainfall data.

Dataset: https://www.kaggle.com/datasets/ggopinathan/rainfall-in-kerala-from-19012017

---

**Project Status:** Research implementation, evaluation and reproducibility checks completed. Final documentation and submission review in progress.