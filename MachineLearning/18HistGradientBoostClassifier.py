from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold
import optuna

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1.0)
    max_iter = trial.suggest_int('max_iter', 50, 500)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 50)
    max_depth = trial.suggest_int('max_depth', 1, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 50)
    l2_regularization = trial.suggest_float('l2_regularization', 0.0, 10.0)
    max_features = trial.suggest_float('max_features', 0.1, 1.0)

    HGBCHyper = HistGradientBoostingClassifier(learning_rate=learning_rate,
                                               max_iter=max_iter,
                                               max_leaf_nodes=max_leaf_nodes,
                                               max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf,
                                               l2_regularization=l2_regularization,
                                               max_features=max_features)

    # 使用KFold进行10折交叉验证
    kf = KFold(n_splits=10)
    score = cross_val_score(HGBCHyper, x_discovery, y_discovery, cv=kf, scoring='accuracy')
    return score.mean()


# 创建Optuna优化器
study = optuna.create_study(direction='maximize')

# 进行10次优化
for _ in range(10):
    study.optimize(objective, n_trials=100)

print(study.best_params)

best_params = study.best_params

HGBC = HistGradientBoostingClassifier(learning_rate=best_params['learning_rate'],
                                      max_iter=best_params['max_iter'],
                                      max_leaf_nodes=best_params['max_leaf_nodes'],
                                      max_depth=best_params['max_depth'],
                                      min_samples_leaf=best_params['min_samples_leaf'],
                                      l2_regularization=best_params['l2_regularization'],
                                      max_features=best_params['max_features'])

HGBC.fit(x_discovery, y_discovery)

print(f"Accuracy Discovery: {HGBC.score(x_discovery, y_discovery)}")
print(f"Accuracy Test: {HGBC.score(x_test, y_test)}")

y_pred_discovery = HGBC.predict(x_discovery)
y_pred_test = HGBC.predict(x_test)

from sklearn.metrics import f1_score
f1_discovery = f1_score(y_discovery, y_pred_discovery, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')
print(f"F1 Score Discovery: {f1_discovery}")
print(f"F1 Score Test: {f1_test}")


from sklearn.metrics import roc_auc_score
# 预测概率
y_probs_discovery = HGBC.predict_proba(x_discovery)[:, 1]
y_probs_test = HGBC.predict_proba(x_test)[:, 1]

# 计算AUC-ROC分数
roc_auc_discovery = roc_auc_score(y_discovery, y_probs_discovery)
roc_auc_test = roc_auc_score(y_test, y_probs_test)
print(f"AUC-ROC Discovery: {roc_auc_discovery}")
print(f"AUC-ROC Test: {roc_auc_test}")
