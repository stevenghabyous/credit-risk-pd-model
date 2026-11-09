# Credit Risk: Probability of Default Model

An end-to-end probability of default (PD) model that predicts whether a borrower will become 90+ days delinquent within two years, built on 150,000 consumer loan records. The project compares an interpretable logistic regression benchmark with an XGBoost challenger, then turns model scores into a credit acceptance strategy.

## Highlights

- **Data cleaning:** missing-value flags for income and dependents (missing income turned out to be *lower* risk), removal of impossible ages and 96/98 error codes, and 99th percentile winsorizing of heavy-tailed ratios
- **Preprocessing:** stratified 70/30 holdout with the scaler fit on training data only
- **Models:** logistic regression with class weighting, and XGBoost with `scale_pos_weight` for the 6.7% default rate
- **Evaluation:** ROC-AUC, KS, and Gini, plus coefficient and odds-ratio interpretation
- **Credit strategy:** simulated approval cutoffs to show the trade-off between acceptance rate and portfolio default rate

## Results (holdout set)

| Model | ROC-AUC | KS | Gini |
|---|---|---|---|
| Logistic regression | 0.853 | 0.55 | 0.71 |
| XGBoost | 0.858 | 0.56 | 0.72 |

Revolving utilization and past delinquency were the strongest risk drivers in both models. Using XGBoost scores, approving about 81% of applicants cuts the default rate among approved borrowers from 6.6% to 2.5%.

## Limitations and next steps

- Winsorizing caps are computed before the train/test split; computing them on training data only would be stricter.
- Results use a single holdout and fixed XGBoost settings. Next steps: stratified k-fold cross-validation, hyperparameter tuning, probability calibration, and SHAP explanations.

## How to run

1. Download `cs-training.csv` from the [Give Me Some Credit competition on Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it in a `data/` folder in this repository. The data is not included here because of Kaggle's competition rules.
2. Install dependencies: `pip install -r requirements.txt`
3. Open `Credit_Risk_PD_Model.ipynb` in Jupyter and run all cells.

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn
