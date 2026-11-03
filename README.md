# Machine Learning

A hands-on collection of classical machine learning algorithms implemented in Jupyter notebooks with **scikit-learn**, plus an end-to-end regression project deployed as a **Flask** web app.

Each folder covers one algorithm or technique: the intuition, a working implementation on real or synthetic data, evaluation, and (where relevant) hyperparameter tuning.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Supervised Learning: Regression](#supervised-learning-regression)
- [Supervised Learning: Classification](#supervised-learning-classification)
- [Ensemble Methods](#ensemble-methods)
- [Unsupervised Learning: Clustering](#unsupervised-learning-clustering)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Anomaly Detection](#anomaly-detection)
- [End-to-End Project: Algerian Forest Fires](#end-to-end-project-algerian-forest-fires)
- [Datasets](#datasets)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)

---

## Repository Structure

```
Machine-Learning/
├── Linear Regression Scratch/   # Linear regression from scratch (Gradient Descent & OLS)
├── Linear Regression/           # Simple, multiple & polynomial regression
├── Lasso Regression/            # L1 regularization
├── Logistic Regression/         # Binary, multiclass, imbalanced data, ROC-AUC
├── Naive Bayes/                 # Gaussian Naive Bayes
├── KNN/                         # K-Nearest Neighbors classifier & regressor
├── SVM/                         # SVC, SVR, kernels
├── Decision Tree/               # Classifier, regressor, pruning
├── Random Forest/               # Bagging, Random Forest, Travel package prediction
├── Gradient Boosting/           # GradientBoosting classifier & regressor
├── AdaBoost/                    # AdaBoost classifier & regressor
├── XGBoost/                     # XGBoost classifier
├── Ensemble Heterogenous/       # Voting & Stacking
├── K-Means/                     # K-Means, Elbow & Silhouette methods
├── Heirarchical Clustering/     # Agglomerative clustering
├── DBSCAN/                      # Density-based clustering
├── PCA/                         # Principal Component Analysis
├── Anomaly Detection/           # Isolation Forest, LOF, DBSCAN
├── Algerian Fire Project/       # EDA + model training (forest fire FWI prediction)
└── Flask Deployment/            # Flask web app serving the trained Ridge model
```

---

## Supervised Learning: Regression

| Folder | Notebook | What it covers |
|---|---|---|
| **Linear Regression Scratch** | `code.ipynb` | Linear regression class built from scratch with NumPy using **gradient descent** |
| | `OLS-Lr.ipynb` | Linear regression from scratch using the **Ordinary Least Squares** closed-form solution |
| **Linear Regression** | `LR.ipynb` | Simple linear regression: predicting height from weight, standardization, MAE / MSE / RMSE / R² |
| | `EcoIndx.ipynb` | Multiple linear regression on an economic index dataset with cross-validation |
| | `LinearRegression.ipynb` | Linear regression on the medical insurance cost dataset |
| | `PolynomialRegression.ipynb` | Polynomial features for non-linear (quadratic) data, compared with plain linear regression |
| **Lasso Regression** | `LassoRegression.ipynb` | Lasso (L1) and `LassoCV` on the insurance dataset |
| **SVM** | `svmRegressor.ipynb` | `SVR` and `LinearSVR` on the diabetes dataset with `GridSearchCV` |
| | `SVR.ipynb` | SVR on the `tips` dataset with label/one-hot encoding and `ColumnTransformer` |
| **KNN** | `KNNRegressor.ipynb` | `KNeighborsRegressor` on synthetic regression data |
| **Decision Tree** | `DtRegressor.ipynb` | `DecisionTreeRegressor` on the diabetes dataset, tree visualization, comparison with Random Forest |

## Supervised Learning: Classification

| Folder | Notebook | What it covers |
|---|---|---|
| **Logistic Regression** | `logisticReg.ipynb` | Heart disease prediction with scaling, confusion matrix and classification report |
| | `LogReg.ipynb` | Hyperparameter tuning with `GridSearchCV`, `RandomizedSearchCV` and `StratifiedKFold` |
| | `LogRegMultiClass.ipynb` | Multiclass logistic regression |
| | `LogRegImbalance.ipynb` | Handling **imbalanced datasets** (class weights + grid search) |
| | `ROC-AUC.ipynb` | ROC curve and ROC-AUC score, choosing a decision threshold |
| **Naive Bayes** | `naiveBayes.ipynb` | `GaussianNB` on the heart disease dataset |
| | `NB.ipynb` | Naive Bayes on Iris, plus one-hot encoding on `tips` (`get_dummies` vs `OneHotEncoder`) |
| **KNN** | `knn.ipynb` | KNN on heart disease data: why scaling matters, `GridSearchCV`, and `Pipeline` |
| | `KNNClassifier.ipynb` | `KNeighborsClassifier` on synthetic data with tuning |
| **SVM** | `svmClassifier.ipynb` | `SVC` on the Iris dataset |
| | `svmC2.ipynb` | SVC with `GridSearchCV` over `C`, `gamma` and kernels |
| | `svmKernel.ipynb` | In-depth intuition for SVM **kernels** (polynomial / RBF) on non-linearly separable data |
| **Decision Tree** | `DtClassifier.ipynb` | Titanic survival: imputation, encoding, **pre-pruning** and **post-pruning** |
| | `DtC2.ipynb` | Decision tree on Iris with `GridSearchCV` and tree plotting |

## Ensemble Methods

| Folder | Notebook | What it covers |
|---|---|---|
| **Random Forest** | `randomForest.ipynb` | Bagging (Logistic Regression & Decision Tree base learners) and `RandomForestClassifier` on Titanic |
| | `travel.ipynb` | End-to-end **Holiday Package prediction**: data cleaning, imputation, feature engineering, `ColumnTransformer`, model comparison, `RandomizedSearchCV`, ROC-AUC |
| **Gradient Boosting** | `GB.ipynb` | `GradientBoostingClassifier` and `GradientBoostingRegressor` |
| **AdaBoost** | `adaBoostClassi.ipynb` | `AdaBoostClassifier` with a decision-stump base learner |
| | `adaBoostRegres.ipynb` | `AdaBoostRegressor` with MAE / MSE / R² evaluation |
| **XGBoost** | `xgBoost.ipynb` | XGBoost classifier with classification report |
| **Ensemble Heterogenous** | `voting.ipynb` | `VotingClassifier` and `VotingRegressor` (Logistic/Linear, SVM, Decision Tree) |
| | `stacking.ipynb` | `StackingClassifier` and `StackingRegressor` |

## Unsupervised Learning: Clustering

| Folder | Notebook | What it covers |
|---|---|---|
| **K-Means** | `KMeans.ipynb` | K-Means basics and choosing *k* with the **Elbow** and **Silhouette** methods |
| | `kMeans2.ipynb` | K-Means with scaling and silhouette evaluation |
| | `irisKMeans.ipynb` | K-Means on Iris, visualized with PCA |
| **Heirarchical Clustering** | `heirarchical.ipynb` | Agglomerative clustering and dendrograms on Iris |
| **DBSCAN** | `dbscan.ipynb` | DBSCAN on Iris |
| | `nonLinearDBSCAN.ipynb` | DBSCAN vs K-Means on non-linear (moon-shaped) data |

## Dimensionality Reduction

| Folder | Notebook | What it covers |
|---|---|---|
| **PCA** | `pca.ipynb` | PCA on Iris and explained variance |
| | `pca2.ipynb` | PCA on the Breast Cancer dataset (30 features down to 2 components) |

## Anomaly Detection

| Folder | Notebook | What it covers |
|---|---|---|
| **Anomaly Detection** | `isolationForest.ipynb` | `IsolationForest` on the thyroid dataset, visualized with PCA |
| | `LOF.ipynb` | `LocalOutlierFactor` on the thyroid dataset |
| | `dbscan.ipynb` | Using DBSCAN noise points as outliers on moon-shaped data |

---

## End-to-End Project: Algerian Forest Fires

Predicts the **Fire Weather Index (FWI)** from weather readings in two Algerian regions (Bejaia and Sidi Bel-Abbes).

**1. EDA & model training** (`Algerian Fire Project/`)
- `EDA.ipynb`: data cleaning, splitting the raw file by region, fixing data types, correlation analysis, and month-wise fire analysis (August and September have the most fires in both regions).
- `modelTraining.ipynb`: feature selection, `StandardScaler`, then Linear, **Lasso**, **Ridge** and **ElasticNet** regression, with cross-validated variants (`LassoCV`, `RidgeCV`, `ElasticNetCV`). The Ridge model and the scaler are saved as `ridge.pkl` and `scaler.pkl`.

**2. Deployment** (`Flask Deployment/`)

A Flask app that loads the pickled Ridge model and scaler and predicts FWI from a web form.

```
Flask Deployment/
├── application.py        # Flask app (routes: / and /predictdata)
├── models/               # ridge.pkl, scaler.pkl
├── notebooks/            # EDA + training notebooks
├── templates/            # index.html, home.html
└── requirements.txt
```

**Input features:** Temperature, RH (relative humidity), Ws (wind speed), Rain, FFMC, DMC, ISI, Classes, Region

Run it locally:

```bash
cd "Flask Deployment"
pip install flask numpy pandas scikit-learn
python application.py
```

Then open http://localhost:5000/predictdata in your browser.

---

## Datasets

| Dataset | Source | Used in |
|---|---|---|
| Heart disease (`heart.csv`) | Included | Logistic Regression, KNN, Naive Bayes |
| Insurance (`insurance.csv`) | Included | Linear Regression, Lasso Regression |
| Height-Weight (`height-weight.csv`) | Included | Linear Regression |
| Economic Index (`economic_index.csv`) | Included | Linear Regression |
| Travel / Holiday Package (`Travel.csv`) | Included | Random Forest |
| Algerian Forest Fires | Included | Algerian Fire Project, Flask Deployment |
| Thyroid (`thyroid_dataset.csv`) | Included | Anomaly Detection |
| Iris, Diabetes, Breast Cancer | `sklearn.datasets` | SVM, Decision Tree, Naive Bayes, K-Means, DBSCAN, PCA, Clustering |
| Titanic, Tips | `seaborn.load_dataset` | Decision Tree, Random Forest, Naive Bayes, SVM |
| Synthetic data | `make_classification`, `make_regression`, `make_blobs`, `make_moons` | Most algorithm demos |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/md-zaid-10/Machine-Learning.git
cd Machine-Learning

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter flask

# Launch Jupyter
jupyter notebook
```

Open any folder and run its notebooks top to bottom. Notebooks that use a local CSV expect to be run from inside their own folder.

---

## Tech Stack

- **Language:** Python 3
- **Data:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn, XGBoost
- **Deployment:** Flask
- **Environment:** Jupyter Notebook

---

## Author

**Md Zaid** · [GitHub @md-zaid-10](https://github.com/md-zaid-10)

If this repo helps you, consider giving it a star.
