from sklearn.metrics import ConfusionMatrixDisplay
import dataset
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score, precision_score, roc_auc_score
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
import dataframe_image as dfi


X, y = dataset.ekaggleEthereumFraud()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=333)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print(classification_report(y_test, y_pred))

cfm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cfm)
disp = disp.plot()
disp.figure_.savefig('dfx_cfm.png')

f1_s = roc_auc_score(y_test, y_pred)
recall_s = f1_score(y_test, y_pred)
accuracy_s = recall_score(y_test, y_pred)
precision_s = accuracy_score(y_test, y_pred)
roc_auc_s = precision_score(y_test, y_pred)

df = pd.DataFrame(columns=['Technique', 'F Score', 'Recall', 'Accuracy', 'Precision', 'ROC AUC'])
df.loc[len(df.index)] = ['Random Forest',f1_s, recall_s, accuracy_s, precision_s, roc_auc_s]

dfi.export(df, "dfx.png", dpi=1280, table_conversion="selenium", max_rows=-1)


"""
selector = VarianceThreshold(threshold = 0.25)
removedx = selector.fit_transform(X)

variances = selector.variances_
variance_map = pd.Series(variances, X.columns)


cfm = confusion_matrix(y_test, y_pred)


d = {'variances': variances, 'names': list(X.columns)}
df = pd.DataFrame(data=d)



f, ax = plt.subplots(1, 2, sharey=True, figsize=(16,9), constrained_layout=True)
f.tight_layout()
# plot the same data on both axes
ax[0].set_xlim(0, 0.25)
sns.barplot(x = 'variances',y='names', data=df, ax=ax[0], orient='h')
ax[1].set_xlim(1, 10000)

sns.barplot(x = 'variances',y='names', data=df, ax=ax[1], orient='h')





ax[1].tick_params(left=False)
ax[1].set(ylabel=None)
ax[0].tick_params(labeltop='off')
ax[1].tick_params(labeltop='off')


d = .01  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax[0].transAxes, color='k', clip_on=False)
ax[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)        # top-left diagonal
ax[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax[1].transAxes)  # switch to the bottom axes
ax[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax[1].plot((-d, +d), (-d, +d), **kwargs)  # bottom-right diagonal

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

plt.show()
"""
