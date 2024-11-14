from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']
# Fit and transform the target labels
y_discovery = le.fit_transform(y_discovery)

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']
y_test = le.fit_transform(y_test)

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
import optuna

def objective(trial):
    booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])

    param = {
        "booster": booster,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10),
        "objective": 'binary:logistic',
    }

    if booster == "gbtree" or booster == "dart":
        # 仅适用于树模型的超参数
        param['max_depth'] = trial.suggest_int('max_depth', 1, 10)
        param['max_leaves'] = trial.suggest_int('max_leaves', 0, 30)
        param['gamma'] = trial.suggest_float('gamma', 0.0, 0.5)

    XGBHyper = XGBClassifier(**param)

    # 使用KFold进行10折交叉验证
    kf = KFold(n_splits=10)
    score = cross_val_score(XGBHyper, x_discovery, y_discovery, cv=kf, scoring='accuracy')
    return score.mean()


# 创建Optuna优化器
study = optuna.create_study(direction='maximize')

# 进行10次优化
for _ in range(10):
    study.optimize(objective, n_trials=100)

print(study.best_params)

best_params = study.best_params

XGB = XGBClassifier(**best_params)

XGB.fit(x_discovery, y_discovery)

print(f"Accuracy Discovery: {XGB.score(x_discovery, y_discovery)}")
print(f"Accuracy Test: {XGB.score(x_test, y_test)}")

y_pred_discovery = XGB.predict(x_discovery)
y_pred_test = XGB.predict(x_test)

from sklearn.metrics import f1_score
f1_discovery = f1_score(y_discovery, y_pred_discovery, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')
print(f"F1 Score Discovery: {f1_discovery}")
print(f"F1 Score Test: {f1_test}")

from sklearn.metrics import roc_auc_score
# 预测概率
y_prob_discovery = XGB.predict_proba(x_discovery)[:, 1]
y_prob_test = XGB.predict_proba(x_test)[:, 1]

# 计算AUC-ROC分数
roc_auc_discovery = roc_auc_score(y_discovery, y_prob_discovery)
roc_auc_test = roc_auc_score(y_test, y_prob_test)
print(f"AUC-ROC Discovery: {roc_auc_discovery}")
print(f"AUC-ROC Test: {roc_auc_test}")
