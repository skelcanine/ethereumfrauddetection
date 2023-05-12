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
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import imblearn
import dataframe_image as dfi
from sklearn.metrics import ConfusionMatrixDisplay




"""
call_metric = getattr(sklearn.metrics, score)
metric_result = call_metric(y_test, y_pred)


"""

X, y = dataset.ekaggleEthereumFraud()

smote = SMOTE()

X, y = smote.fit_resample(X, y)

enn = EditedNearestNeighbours()

X, y = enn.fit_resample(X, y)

""" ANOTHER ? FEATURE SELECTION
importances = mutual_info_classif(X, y)
print(importances)

feat_importances = pd.Series(importances, X.columns)
feat_importances.plot(kind='barh', color = 'teal')
plt.show()
"""
selector = VarianceThreshold(threshold = 0.25)
removedx = selector.fit_transform(X)
print(selector.get_feature_names_out())
variances = selector.variances_
print(variances)
print(selector.get_support())

variance_map = pd.Series(variances, X.columns)
variance_map.plot(kind='barh', color='teal')

for i, v in enumerate(variances):
    plt.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')

plt.show(dpi =1500)



concol = [column for column in X.columns
          if column not in X.columns[selector.get_support()]]

for features in concol:
    print(features)

# X.drop(concol, axis=1)
print(removedx.shape)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

scaler = MinMaxScaler()

X_scaled_minmax = scaler.fit_transform(removedx)

scaler = RobustScaler()

X_scaled_robust = scaler.fit_transform(removedx)

scaler = StandardScaler()

X_scaled_variance = scaler.fit_transform(removedx)

pca = PCA(n_components=38)
X_scaled_pca = pca.fit_transform(X_scaled)


pca = PCA(n_components=30)
X_pca = pca.fit_transform(X)

print(X_pca)

pca = PCA(n_components=38)
x = pca.fit(X_scaled)
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

"""
px.area(
    x=range(1, exp_var_cumul.shape[0] + 1),
    y=exp_var_cumul,
    labels={"x": "# Components", "y": "Explained Variance"}
).show()
"""
print(X_scaled.shape)

SEED = 112
"""
print("@@@ CLASSIC")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))

cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("@@@ VARIANCE")
X_train, X_test, y_train, y_test = train_test_split(removedx, y, test_size=0.3, random_state=SEED)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))

cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("@@@ SCALED")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=SEED)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))

cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("@@@ SCALEDMINMAX")
X_train, X_test, y_train, y_test = train_test_split(X_scaled_minmax, y, test_size=0.3, random_state=SEED)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))

cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("@@@ SCALEDrobust")
X_train, X_test, y_train, y_test = train_test_split(X_scaled_robust, y, test_size=0.3, random_state=SEED)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))

cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("@@@ SCALED VARIANCE")
X_train, X_test, y_train, y_test = train_test_split(X_scaled_variance, y, test_size=0.3, random_state=SEED)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))

cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("@@@ PCA")
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=SEED)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))

cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("@@@ SCALED PCA")
X_train, X_test, y_train, y_test = train_test_split(X_scaled_pca, y, test_size=0.3, random_state=SEED)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))

cfm = confusion_matrix(y_test, y_pred)
print(cfm)
"""


def testdatasetFunction(X, y, namee):
    cfmtotal = 0
    if not type(X) == type(y):
        X = pd.DataFrame(X)
    f1 = list()
    recall = list()
    accuracy = list()
    precision = list()
    roc_auc = list()
    divider = 15*20

    for j in range(20):
        strtfdKFold = RepeatedStratifiedKFold(n_splits=5,n_repeats=3, random_state=SEED)
        kfold = strtfdKFold.split(X, y)
        for i, (train,test) in enumerate(kfold):
            X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
            rfc = RandomForestClassifier()
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)

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
                cfmtotal = cfmtotal+ cfm

    print("@@@@@@",namee,"@@@@@@@")
    print("f1. score mean= ", sum(f1)/divider)
    print("recall score mean= ", sum(recall)/divider)
    print("accuracy score mean= ", sum(accuracy)/divider)
    print("precision score mean= ", sum(precision)/divider)
    print("roc_auc score mean= ", sum(roc_auc)/divider)
    print(cfmtotal/divider)


"""
TEST FOR BEST SCALING AND FEATURE SELECTION
testdatasetFunction(X, y,"X")
testdatasetFunction(removedx, y, "variance")
testdatasetFunction(X_scaled, y, "X_scaled")
testdatasetFunction(X_scaled_variance, y, "X_scaled_variance")
testdatasetFunction(X_scaled_minmax, y, "X_scaled_minmax")
testdatasetFunction(X_scaled_robust, y, "X_scaled_robust")
"""

""" PARAMETER FINDING
param_grid = {
    'n_estimators': [225, 235, 245, 215],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [30, 35, 33, 37],
    'max_leaf_nodes': [190, 192, 194],
    'n_jobs': [-1]
}

grid_search = GridSearchCV(RandomForestClassifier(max_depth=35, max_leaf_nodes=190, n_estimators=225, n_jobs=-1),
                           param_grid=param_grid)
grid_search.fit(X_train, y_train)


y_pred_grid = grid_search.predict(X_test)

print("f1. score ",    roc_auc_score(y_test, y_pred_grid))
print("recall score = ", f1_score(y_test, y_pred_grid))
print("accuracy score = ", recall_score(y_test, y_pred_grid))
print("precision score = ", accuracy_score(y_test, y_pred_grid))
print("roc_auc score = ", precision_score(y_test, y_pred_grid))
print(confusion_matrix(y_test, y_pred_grid))


print(grid_search.best_estimator_)


X_train, X_test, y_train, y_test = train_test_split(X_scaled_variance, y, test_size=0.3, random_state=SEED)
rfc = RandomForestClassifier(max_depth=35, max_leaf_nodes=190, n_estimators=225, n_jobs=-1)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print("f1. score ", roc_auc_score(y_test, y_pred))
print("recall score = ", f1_score(y_test, y_pred))
print("accuracy score = ", recall_score(y_test, y_pred))
print("precision score = ", accuracy_score(y_test, y_pred))
print("roc_auc score = ", precision_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

"""

df = pd.DataFrame(columns=['Technique', 'F Score', 'Recall', 'Accuracy', 'Precision', 'ROC AUC',
                           'Recall nonsampled', 'Accuracy nonsampled', 'ROC AUC nonsampled'])

cfmdict = dict()

undersamplers = ['EditedNearestNeighbours', 'RandomUnderSampler', 'TomekLinks', 'NearMiss']
oversamplers = ['ADASYN', 'RandomOverSampler', 'SMOTE']

X, y = dataset.ekaggleEthereumFraud()

selector = VarianceThreshold(threshold=0.25)
X = selector.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train_nonsampled, X_test_nonsampled, y_train_nonsampled, y_test_nonsampled = train_test_split(X, y, test_size=0.3, random_state=SEED)

for oversampler in oversamplers:
    call_oversampler = getattr(imblearn.over_sampling, oversampler)
    myoversampler = call_oversampler()
    X_oversam, y_oversam = myoversampler.fit_resample(X, y)
    for undersampler in undersamplers:
        call_undersampler = getattr(imblearn.under_sampling, undersampler)
        myundersampler = call_undersampler()
        X_res, y_res = myundersampler.fit_resample(X_oversam, y_oversam)



        f1 = list()
        recall = list()
        accuracy = list()
        precision = list()
        roc_auc = list()
        recall_nonsampled = list()
        accuracy_nonsampled = list()
        roc_auc_nonsampled = list()

        cfmtotal = 0
        cfmnonstotal = 0
        divider = 10*5


        strtfdKFold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=SEED)
        kfold = strtfdKFold.split(X_res, y_res)
        for i, (train, test) in enumerate(kfold):
            X_train, X_test, y_train, y_test = X_res[train], X_res[test], y_res[train], y_res[test]
            rfc = RandomForestClassifier(max_depth=35, max_leaf_nodes=190, n_estimators=225, n_jobs=-1)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            y_pred_nonsampled = rfc.predict(X_test_nonsampled)

            f1_s = roc_auc_score(y_test, y_pred)
            recall_s = f1_score(y_test, y_pred)
            accuracy_s = recall_score(y_test, y_pred)
            precision_s = accuracy_score(y_test, y_pred)
            roc_auc_s = precision_score(y_test, y_pred)

            recall_nonsampled_s = f1_score(y_test_nonsampled, y_pred_nonsampled)
            accuracy_nonsampled_s = accuracy_score(y_test_nonsampled, y_pred_nonsampled)
            roc_auc_nonsampled_s = precision_score(y_test_nonsampled, y_pred_nonsampled)

            f1.append(f1_s)
            recall.append(recall_s)
            accuracy.append(accuracy_s)
            precision.append(precision_s)
            roc_auc.append(roc_auc_s)

            recall_nonsampled.append(recall_nonsampled_s)
            accuracy_nonsampled.append(accuracy_nonsampled_s)
            roc_auc_nonsampled.append(roc_auc_nonsampled_s)

            cfm = confusion_matrix(y_test, y_pred)
            cfmnons = confusion_matrix(y_test, y_pred)

            if i == 0:
                cfmtotal = cfm
                cfmnonstotal = cfmnons
            else:
                cfmtotal = cfmtotal + cfm
                cfmnonstotal = cfmnonstotal + cfmnons

        name = oversampler + " THEN " + undersampler
        cfmdict[name] = cfmnonstotal / divider
        print("@@@@@@@", name, "@@@@@@@")
        f1_mean = sum(f1) / divider
        recall_mean = sum(recall) / divider
        accuracy_mean = sum(accuracy) / divider
        precision_mean = sum(precision) / divider
        roc_auc_mean = sum(roc_auc) / divider
        recall_nonsampled_mean = sum(recall_nonsampled) / divider
        accuracy_nonsampled_mean = sum(accuracy_nonsampled) / divider
        roc_auc_nonsampled_mean = sum(roc_auc_nonsampled) / divider
        print("f1. score mean= ", f1_mean)
        print("recall score mean= ", recall_mean)
        print("accuracy score mean= ", accuracy_mean)
        print("precision score mean= ", precision_mean)
        print("roc_auc score mean= ", roc_auc_mean)
        print("recall_nonsampled score mean= ", recall_nonsampled_mean)
        print("accuracy_nonsampled score mean= ", accuracy_nonsampled_mean)
        print("roc_auc_nonsampled score mean= ", roc_auc_nonsampled_mean)

        df.loc[len(df.index)] = [name, f1_mean, recall_mean, accuracy_mean, precision_mean, roc_auc_mean, recall_nonsampled_mean, accuracy_nonsampled_mean, roc_auc_nonsampled_mean]
        print(cfmtotal / divider)

X, y = dataset.ekaggleEthereumFraud()

selector = VarianceThreshold(threshold=0.25)
X = selector.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

for undersampler in undersamplers:
    call_undersampler = getattr(imblearn.under_sampling, undersampler)
    myundersampler = call_undersampler()
    X_undersam, y_undersam = myundersampler.fit_resample(X, y)
    for oversampler in oversamplers:
        call_oversampler = getattr(imblearn.over_sampling, oversampler)
        myoversampler = call_oversampler()
        X_res, y_res = myoversampler.fit_resample(X_undersam, y_undersam)



        f1 = list()
        recall = list()
        accuracy = list()
        precision = list()
        roc_auc = list()
        recall_nonsampled = list()
        accuracy_nonsampled = list()
        roc_auc_nonsampled = list()

        cfmtotal = 0
        cfmnonstotal =0
        divider = 10*5


        strtfdKFold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=SEED)
        kfold = strtfdKFold.split(X_res, y_res)
        for i, (train, test) in enumerate(kfold):
            X_train, X_test, y_train, y_test = X_res[train], X_res[test], y_res[train], y_res[test]
            rfc = RandomForestClassifier(max_depth=35, max_leaf_nodes=190, n_estimators=225, n_jobs=-1)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            y_pred_nonsampled = rfc.predict(X_test_nonsampled)

            f1_s = roc_auc_score(y_test, y_pred)
            recall_s = f1_score(y_test, y_pred)
            accuracy_s = recall_score(y_test, y_pred)
            precision_s = accuracy_score(y_test, y_pred)
            roc_auc_s = precision_score(y_test, y_pred)

            recall_nonsampled_s = f1_score(y_test_nonsampled, y_pred_nonsampled)
            accuracy_nonsampled_s = accuracy_score(y_test_nonsampled, y_pred_nonsampled)
            roc_auc_nonsampled_s = precision_score(y_test_nonsampled, y_pred_nonsampled)

            f1.append(f1_s)
            recall.append(recall_s)
            accuracy.append(accuracy_s)
            precision.append(precision_s)
            roc_auc.append(roc_auc_s)

            recall_nonsampled.append(recall_nonsampled_s)
            accuracy_nonsampled.append(accuracy_nonsampled_s)
            roc_auc_nonsampled.append(roc_auc_nonsampled_s)

            cfm = confusion_matrix(y_test, y_pred)
            cfmnons = confusion_matrix(y_test, y_pred_nonsampled)

            if i == 0:
                cfmtotal = cfm
                cfmnonstotal = cfmnons
            else:
                cfmtotal = cfmtotal + cfm
                cfmnonstotal = cfmnonstotal + cfmnons


        name = undersampler + " THEN " + oversampler
        cfmdict[name]=cfmnonstotal/divider
        print("@@@@@@@", name, "@@@@@@@")
        f1_mean = sum(f1) / divider
        recall_mean = sum(recall) / divider
        accuracy_mean = sum(accuracy) / divider
        precision_mean = sum(precision) / divider
        roc_auc_mean = sum(roc_auc) / divider
        recall_nonsampled_mean = sum(recall_nonsampled) / divider
        accuracy_nonsampled_mean = sum(accuracy_nonsampled) / divider
        roc_auc_nonsampled_mean = sum(roc_auc_nonsampled) / divider
        print("f1. score mean= ", f1_mean)
        print("recall score mean= ", recall_mean)
        print("accuracy score mean= ", accuracy_mean)
        print("precision score mean= ", precision_mean)
        print("roc_auc score mean= ", roc_auc_mean)
        print("recall_nonsampled score mean= ", recall_nonsampled_mean)
        print("accuracy_nonsampled score mean= ", accuracy_nonsampled_mean)
        print("roc_auc_nonsampled score mean= ", roc_auc_nonsampled_mean)

        df.loc[len(df.index)] = [name, f1_mean, recall_mean, accuracy_mean, precision_mean, roc_auc_mean,
                                 recall_nonsampled_mean, accuracy_nonsampled_mean, roc_auc_nonsampled_mean]
        print(cfmtotal / divider)

pd.set_option('display.max_columns', None)
df = df.sort_values(by=['Recall nonsampled', 'Recall'], ascending=False)
print(df.head(10))
print(df.tail(10))
df = df.head(10)


#dfi.export(df, "df.png", dpi=1280, table_conversion="selenium", max_rows=-1)
#df_style = df.style.background_gradient(subset='Recall nonsampled')
#dfi.export(df_style, "df_styled.png", dpi=1280, table_conversion="selenium", max_rows=-1)

for index, row in df.iterrows():
    cfname = row['Technique']
    cm = cfmdict[cfname]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp = disp.plot()
    disp.figure_.savefig('cfm/'+cfname+'.png')
    print(cfname)
    print(cfmdict[cfname])

df_indexed = df.set_index('Technique')
#dfi.export(df_indexed, "df_indexed.png", dpi=1280, table_conversion="selenium", max_rows=-1)
df_indexed_style = df_indexed.style.background_gradient(subset='Recall nonsampled')
dfi.export(df_indexed_style, "df_indexed_styled.png", dpi=1280, table_conversion="selenium", max_rows=-1)



