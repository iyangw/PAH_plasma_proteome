from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.loc[:, ['TTR', 'CRISP3']]
y_discovery = discovery.loc[:, 'Condition']

x_test = test.loc[:, ['TTR', 'CRISP3']]
y_test = test.loc[:, 'Condition']

from sklearn.neighbors import KNeighborsClassifier

best_params = {'n_neighbors': 11, 'weights': 'uniform', 'metric': 'canberra'}

KNeigh = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'],
                              weights=best_params['weights'],
                              metric=best_params['metric'])

KNeigh.fit(x_discovery, y_discovery)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_discovery = le.fit_transform(y_discovery)
y_test = le.fit_transform(y_test)

y_probs_discovery = KNeigh.predict_proba(x_discovery)[:, 1]
y_probs_test = KNeigh.predict_proba(x_test)[:, 1]

fpr_discovery, tpr_discovery, _ = roc_curve(y_discovery, y_probs_discovery)
auc_discovery = roc_auc_score(y_discovery, y_probs_discovery)

fpr_test, tpr_test, _ = roc_curve(y_test, y_probs_test)
auc_test = roc_auc_score(y_test, y_probs_test)

# 绘制ROC曲线
plt.figure(figsize=(8, 6), dpi=600)
plt.rcParams['font.family'] = ["Arial"]
plt.plot(fpr_discovery, tpr_discovery, label=f"Discovery Set, AUC={auc_discovery:.4f}")
plt.plot(fpr_test, tpr_test, label=f"Test Set, AUC={auc_test:.4f}")

plt.plot([0, 1], [0, 1], 'k--', lw=1)  # 随机猜测的对角线
plt.xlabel("False Positive Rate", fontsize=14, labelpad=15)
plt.ylabel("True Positive Rate", fontsize=14, labelpad=15)
plt.title("ROC Curve")
plt.legend(loc="lower right")
# plt.grid()
plt.show()
