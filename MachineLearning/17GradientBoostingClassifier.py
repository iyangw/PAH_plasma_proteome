from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_depth = trial.suggest_int('max_depth', 1, 10)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 50)
    max_features = trial.suggest_float('max_features', 0.1, 1.0)
    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.5)
    ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.1)

    GBCHyper = GradientBoostingClassifier(n_estimators=n_estimators,
                                          learning_rate=learning_rate,
                                          max_depth=max_depth,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
                                          max_leaf_nodes=max_leaf_nodes,
                                          max_features=max_features,
                                          min_impurity_decrease=min_impurity_decrease,
                                          ccp_alpha=ccp_alpha)

    # 使用KFold进行10折交叉验证
    kf = KFold(n_splits=10)
    score = cross_val_score(GBCHyper, x_discovery, y_discovery, cv=kf, scoring='accuracy')
    return score.mean()


# 创建Optuna优化器
study = optuna.create_study(direction='maximize')

# 进行10次优化
for _ in range(10):
    study.optimize(objective, n_trials=100)

print(study.best_params)

best_params = study.best_params

GBC = GradientBoostingClassifier(n_estimators=best_params['n_estimators'],
                                 learning_rate=best_params['learning_rate'],
                                 max_depth=best_params['max_depth'],
                                 min_samples_split=best_params['min_samples_split'],
                                 min_samples_leaf=best_params['min_samples_leaf'],
                                 max_leaf_nodes=best_params['max_leaf_nodes'],
                                 min_impurity_decrease=best_params['min_impurity_decrease'],
                                 ccp_alpha=best_params['ccp_alpha'],
                                 max_features=best_params['max_features'])


GBC.fit(x_discovery, y_discovery)

print(f"Accuracy Discovery: {GBC.score(x_discovery, y_discovery)}")
print(f"Accuracy Test: {GBC.score(x_test, y_test)}")

y_pred_discovery = GBC.predict(x_discovery)
y_pred_test = GBC.predict(x_test)

from sklearn.metrics import f1_score
f1_discovery = f1_score(y_discovery, y_pred_discovery, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')
print(f"F1 Score Discovery: {f1_discovery}")
print(f"F1 Score Test: {f1_test}")

from sklearn.metrics import roc_auc_score
# 预测概率
y_probs_discovery = GBC.predict_proba(x_discovery)[:, 1]
y_probs_test = GBC.predict_proba(x_test)[:, 1]

# 计算AUC-ROC分数
roc_auc_discovery = roc_auc_score(y_discovery, y_probs_discovery)
roc_auc_test = roc_auc_score(y_test, y_probs_test)
print(f"AUC-ROC Discovery: {roc_auc_discovery}")
print(f"AUC-ROC Test: {roc_auc_test}")
