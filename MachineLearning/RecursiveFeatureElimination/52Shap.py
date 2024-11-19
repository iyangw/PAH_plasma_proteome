from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x = data.loc[:, ['GDI2', 'C7', 'FCGR3A', 'IGKV1-12', 'TTR', 'CRTAC1', 'ICAM3', 'HSPA5', 'PNP', 'RNH1', 'DEFA1', 'ABCA1', 'CRISP3']]
y = data.loc[:, 'Condition']

x_discovery = discovery.loc[:, ['GDI2', 'C7', 'FCGR3A', 'IGKV1-12', 'TTR', 'CRTAC1', 'ICAM3', 'HSPA5', 'PNP', 'RNH1', 'DEFA1', 'ABCA1', 'CRISP3']]
y_discovery = discovery.loc[:, 'Condition']

x_test = test.loc[:, ['GDI2', 'C7', 'FCGR3A', 'IGKV1-12', 'TTR', 'CRTAC1', 'ICAM3', 'HSPA5', 'PNP', 'RNH1', 'DEFA1', 'ABCA1', 'CRISP3']]
y_test = test.loc[:, 'Condition']


from sklearn.neighbors import KNeighborsClassifier

best_params = {'n_neighbors': 18, 'weights': 'distance', 'metric': 'manhattan'}

KNeigh = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'],
                              weights=best_params['weights'],
                              metric=best_params['metric'])

KNeigh.fit(x_discovery, y_discovery)

import shap
# 计算 SHAP 值
explainer = shap.KernelExplainer(KNeigh.predict_proba, x_discovery)
shap_values = explainer.shap_values(x)
print(shap_values[:, :, 1])

import matplotlib.pyplot as plt
plt.figure(dpi=600)
shap.summary_plot(shap_values[:, :, 1], x)



