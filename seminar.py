from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler, TomekLinks, NearMiss
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from sklearn.feature_selection import VarianceThreshold
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

SEED = 112
# Main dataframe



# Get dataset
X, y = dataset.ekaggleEthereumFraud()

# Variance Thresholding
selector = VarianceThreshold(threshold=0.25)
X = selector.fit_transform(X)

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
"""
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=333)



# XGB CLASSIFIER // eta = 0.17, max_depth = 6, n_estimators = 250, scale_pos_weight = 3, subsample = 0.5
print("@@ XGB STARTED")
xgbc = xgb.XGBClassifier()
xgbc.fit(X_train, y_train)
y_predx = xgbc.predict(X_test)

f1_s = roc_auc_score(y_test, y_predx)
recall_s = f1_score(y_test, y_predx)
accuracy_s = recall_score(y_test, y_predx)
precision_s = accuracy_score(y_test, y_predx)
roc_auc_s = precision_score(y_test, y_predx)


print("f1. score mean= ", f1_s)
print("recall score mean= ", recall_s)
print("accuracy score mean= ", accuracy_s)
print("precision score mean= ", precision_s)
print("roc_auc score mean= ", roc_auc_s)

# Logistic Regression // (max_iter=100000, n_jobs=-1, penalty=None, tol=1e-06)
print("@@ LR STARTED")
lr = LogisticRegression(n_jobs=-1, max_iter=100000, penalty=None, tol=1e-06)
lr.fit(X_train, y_train)
y_predl = lr.predict(X_test)

f1_s = roc_auc_score(y_test, y_predl)
recall_s = f1_score(y_test, y_predl)
accuracy_s = recall_score(y_test, y_predl)
precision_s = accuracy_score(y_test, y_predl)
roc_auc_s = precision_score(y_test, y_predl)

print("f1. score mean= ", f1_s)
print("recall score mean= ", recall_s)
print("accuracy score mean= ", accuracy_s)
print("precision score mean= ", precision_s)
print("roc_auc score mean= ", roc_auc_s)

# Random Forest  // max_depth=35, max_leaf_nodes=190, n_estimators=225, n_jobs=-1
print("@@ RF STARTED")
rfc = RandomForestClassifier(max_depth=35, max_leaf_nodes=190, n_estimators=225, n_jobs=-1)
rfc.fit(X_train, y_train)
y_predr = rfc.predict(X_test)

f1_s = roc_auc_score(y_test, y_predr)
recall_s = f1_score(y_test, y_predr)
accuracy_s = recall_score(y_test, y_predr)
precision_s = accuracy_score(y_test, y_predr)
roc_auc_s = precision_score(y_test, y_predr)

print("f1. score mean= ", f1_s)
print("recall score mean= ", recall_s)
print("accuracy score mean= ", accuracy_s)
print("precision score mean= ", precision_s)
print("roc_auc score mean= ", roc_auc_s)

# SVC // C=20, coef0=5, gamma=1, kernel='poly', degree=3
print("@@ SVC STARTED")
svc = svm.SVC(C=20, coef0=5, gamma=1, kernel='poly', degree=3)
svc.fit(X_train, y_train)
y_preds = svc.predict(X_test)

f1_s = roc_auc_score(y_test, y_preds)
recall_s = f1_score(y_test, y_preds)
accuracy_s = recall_score(y_test, y_preds)
precision_s = accuracy_score(y_test, y_preds)
roc_auc_s = precision_score(y_test, y_preds)

print("f1. score mean= ", f1_s)
print("recall score mean= ", recall_s)
print("accuracy score mean= ", accuracy_s)
print("precision score mean= ", precision_s)
print("roc_auc score mean= ", roc_auc_s)

"""

techniquesdict = dict()
techniquesdict['Random Forest'] = RandomForestClassifier(max_depth=35, max_leaf_nodes=190, n_estimators=225, n_jobs=-1)
techniquesdict['Logistic Regression'] = LogisticRegression(n_jobs=-1, max_iter=100000, penalty=None, tol=1e-06)
techniquesdict['SVC'] = svm.SVC()
# C=10, coef0=3, gamma=1, kernel='poly', degree=3
techniquesdict['XGB Classifier'] = xgb.XGBClassifier()

df = pd.DataFrame(columns=['Technique', 'F Score', 'Recall', 'Accuracy', 'Precision', 'ROC AUC'])
cfmdict = dict()

for cname, dictclassifier in techniquesdict.items():
    f1 = list()
    recall = list()
    accuracy = list()
    precision = list()
    roc_auc = list()
    cfmtotal = 0
    cfmnonstotal = 0
    divider = 10 * 5

    print(cname, ":")
    strtfdKFold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=SEED)
    kfold = strtfdKFold.split(X, y)
    for i, (train, test) in enumerate(kfold):
        X_train = X[train]
        X_test, y_train, y_test = X[test], y.iloc[train], y.iloc[test]
        classifier = dictclassifier
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        f1_s = roc_auc_score(y_test, y_pred)
        recall_s = f1_score(y_test, y_pred)
        accuracy_s = recall_score(y_test, y_pred)
        precision_s = accuracy_score(y_test, y_pred)
        roc_auc_s = precision_score(y_test, y_pred)

        f1.append(f1_s)
        recall.append(recall_s)
        accuracy.append(accuracy_s)
        precision.append(precision_s)
        roc_auc.append(roc_auc_s)

        cfm = confusion_matrix(y_test, y_pred)

        if i == 0:
            cfmtotal = cfm

        else:
            cfmtotal = cfmtotal + cfm
            #print(cfmtotal)

    cfmdict[cname] = cfmtotal / divider
    f1_mean = sum(f1) / divider
    recall_mean = sum(recall) / divider
    accuracy_mean = sum(accuracy) / divider
    precision_mean = sum(precision) / divider
    roc_auc_mean = sum(roc_auc) / divider
    df.loc[len(df.index)] = [cname, f1_mean, recall_mean, accuracy_mean, precision_mean, roc_auc_mean]

df = df.sort_values(by=['Recall'], ascending=False)

for index, row in df.iterrows():
    cfname = row['Technique']
    cm = cfmdict[cfname]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp = disp.plot()
    disp.figure_.savefig('seminar/cfm/'+cfname+'.png')
    print(cfname)
    print(cfmdict[cfname])

df_indexed = df.set_index('Technique')
df_indexed_style = df_indexed.style.background_gradient(subset='Recall')
dfi.export(df_indexed_style, "seminar/df_indexed_styled.png", dpi=1280, table_conversion="selenium", max_rows=-1)