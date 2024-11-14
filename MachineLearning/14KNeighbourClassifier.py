from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
import optuna

def objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    metric = trial.suggest_categorical('metric', ['correlation', 'chebyshev', 'cosine',  'manhattan', 'canberra', 'braycurtis', 'euclidean'])


    KNeighHyper = KNeighborsClassifier(n_neighbors=n_neighbors,
                                       weights=weights,
                                       metric=metric)

    # 使用KFold进行10折交叉验证
    kf = KFold(n_splits=10)
    score = cross_val_score(KNeighHyper, x_discovery, y_discovery, cv=kf, scoring='accuracy')
    return score.mean()


# 创建Optuna优化器
study = optuna.create_study(direction='maximize')

# 进行10次优化
for _ in range(10):
    study.optimize(objective, n_trials=100)

print(study.best_params)

best_params = study.best_params

KNeigh = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'],
                              weights=best_params['weights'],
                              metric=best_params['metric'])

KNeigh.fit(x_discovery, y_discovery)

print(f"Accuracy Discovery: {KNeigh.score(x_discovery, y_discovery)}")
print(f"Accuracy Test: {KNeigh.score(x_test, y_test)}")

y_pred_discovery = KNeigh.predict(x_discovery)
y_pred_test = KNeigh.predict(x_test)

from sklearn.metrics import f1_score
f1_discovery = f1_score(y_discovery, y_pred_discovery, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')
print(f"F1 Score Discovery: {f1_discovery}")
print(f"F1 Score Test: {f1_test}")

from sklearn.metrics import roc_auc_score
# 预测概率
y_probs_discovery = KNeigh.predict_proba(x_discovery)[:, 1]
y_probs_test = KNeigh.predict_proba(x_test)[:, 1]

# 计算AUC-ROC分数
roc_auc_discovery = roc_auc_score(y_discovery, y_probs_discovery)
roc_auc_test = roc_auc_score(y_test, y_probs_test)
print(f"AUC-ROC Discovery: {roc_auc_discovery}")
print(f"AUC-ROC Test: {roc_auc_test}")
