# Kerala Flood-Year Prediction Using Machine Learning

## 1. Project Overview

This project investigates whether historical monthly rainfall data can be used to classify documented major flood years in Kerala using classical machine-learning algorithms.

The study explores different rainfall prediction cutoffs to examine how classification performance changes when rainfall information is limited to different months of the year.

**Domain:** Disaster Management  
**SDGs:** SDG 11 – Sustainable Cities and Communities; SDG 13 – Climate Action  
**ML Task:** Binary Classification  
**Article Type:** Short Communication

## 2. Problem Statement

The 2018 Kerala floods caused extensive loss of life and displacement. This project investigates historical flood-year classification using monthly rainfall observations and explores how early in the monsoon a potential warning signal can be identified.

The rainfall dataset covers 1901–2017. Therefore, 2018 is used as contextual motivation and is not included as a model observation.

## 3. Dataset

**Dataset:** Rainfall in Kerala (1901–2017)

**Source:** Kaggle, based on IMD rainfall data.

Dataset link:
https://www.kaggle.com/datasets/ggopinathan/rainfall-in-kerala-from-19012017

The dataset contains 117 annual observations and 19 columns, including monthly rainfall, annual rainfall and seasonal rainfall aggregates.

## 4. Flood-Year Labelling

The target variable is `FLOOD_LABEL`.

- `1`: Year documented as a major flood year in the selected Kerala State Planning Board chronology.
- `0`: Year not listed as a major flood year in that chronology.

Documented positive years:

1924, 1958, 1961, 1992, 2007 and 2013.

Class distribution:

| Class | Observations | Percentage |
|---|---:|---:|
| Documented major flood | 6 | 5.13% |
| Not listed | 111 | 94.87% |

A zero label does not establish that no local flooding occurred in that year.

## 5. Machine-Learning Methods

Five classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbours (k-NN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Random oversampling was used within training folds to address class imbalance.

## 6. Prediction Windows

Rainfall features were evaluated using five prediction cutoffs:

| Cutoff | Available features |
|---|---|
| March | January–March |
| May | January–May |
| June | January–June |
| July | January–July |
| September | January–September |

Only rainfall observations available up to each cutoff were used.

## 7. Evaluation

Models were evaluated using stratified 3-fold cross-validation.

The evaluation metrics were:

- Recall
- F1-score
- ROC-AUC

### Best observed results

| Metric | Model and cutoff | Score |
|---|---|---:|
| Recall | Logistic Regression – June | 0.667 |
| F1-score | Logistic Regression – June | 0.296 |
| ROC-AUC | Logistic Regression – September | 0.737 |

These are exploratory cross-validation results, not independent test-set estimates.

## 8. Project Structure

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
│       └── oof_predictions.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_flood_year_labeling.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── reports/
│   └── figures/
│
├── requirements.txt
└── .gitignore
```

## 9. Installation and Execution

Clone the repository:

```bash
git clone https://github.com/meppyy/ml-project.git
cd ml-project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Execute the notebooks in numerical order, beginning with `01_data_exploration.ipynb`.

## 10. Limitations

- The dataset contains only 117 annual observations.
- Only six years are labelled as documented major flood years under the selected definition.
- The class distribution is highly imbalanced.
- The shuffled cross-validation procedure does not simulate forecasting future years.
- The study does not establish that a model can provide operational flood warnings.
- Regional rainfall aggregates cannot capture every local factor contributing to floods.

The results should therefore be interpreted as an exploratory historical classification study.

## 11. References

1. Government of Kerala, Kerala State Planning Board, *Final Report of the Committee to Examine the Causes of Repeated Extreme Heavy Rainfall Events, Subsequent Floods and Landslides*, December 2019. See Table 5.7 for the selected major flood-year chronology.

2. Kaggle, *Rainfall in Kerala from 1901–2017*.  
   https://www.kaggle.com/datasets/ggopinathan/rainfall-in-kerala-from-19012017

---

**Project status:** Research and implementation completed; final paper preparation pending.