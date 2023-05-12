from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler, TomekLinks, NearMiss
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC

import dataset
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import imblearn
import dataframe_image as dfi
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import svm
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

param_grid = {'penalty': ['l1', 'l2', 'elasticnet', None],  'tol': [1e-6, 1e-7, 1e-5],
              'max_iter': [100000, 200000], 'n_jobs': [-1]}

SEED = 112

# Get dataset
X, y = dataset.ekaggleEthereumFraud()

# Variance Thresholding
selector = VarianceThreshold(threshold=0.25)
X = selector.fit_transform(X)

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=333)

grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)


print(grid.best_estimator_)
print(grid.best_score_)
print(grid.best_params_)