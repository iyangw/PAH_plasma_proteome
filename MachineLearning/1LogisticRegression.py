from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
import optuna

def objective(trial):
    C = trial.suggest_float('C', 1e-6, 100.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'sag', 'saga'])

    # 检查 penalty 和 solver 的兼容性
    if penalty == 'l1' and solver not in ['liblinear', 'saga']:
        raise optuna.exceptions.TrialPruned()  # 'l1' 只与 'liblinear' 或 'saga' 兼容
    if penalty == 'l2' and solver not in ['lbfgs', 'liblinear', 'sag', 'saga']:
        raise optuna.exceptions.TrialPruned()  # 'l2' 适用于所有求解器
    if penalty == 'elasticnet' and solver != 'saga':
        raise optuna.exceptions.TrialPruned()  # 'elasticnet' 仅与 'saga' 兼容
    if penalty is None and solver == 'liblinear':
        raise optuna.exceptions.TrialPruned()  # 'None' 不与 'liblinear' 兼容

    l1_ratio = None
    if penalty == 'elasticnet':
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

    # 创建Logistic Regression模型
    # 如果penalty为None时，忽略C和l1_ratio
    if penalty is None:
        LRHyper = LogisticRegression(penalty=penalty, solver=solver, max_iter=10000)
    else:
        LRHyper = LogisticRegression(C=C, penalty=penalty, solver=solver, l1_ratio=l1_ratio, max_iter=10000)


    # 使用KFold进行10折交叉验证
    kf = KFold(n_splits=10)
    score = cross_val_score(LRHyper, x_discovery, y_discovery, cv=kf, scoring='accuracy')
    return score.mean()


# 创建Optuna优化器
study = optuna.create_study(direction='maximize')

# 进行10次优化
for _ in range(10):
    study.optimize(objective, n_trials=100)

print(study.best_params)

best_params = study.best_params

# 如果penalty为None时，忽略C和l1_ratio
if best_params['penalty'] is None:
    LR = LogisticRegression(penalty=best_params['penalty'], solver=best_params['solver'], max_iter=10000)
else:
    LR = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], solver=best_params['solver'], l1_ratio=best_params.get('l1_ratio', None), max_iter=10000)

LR.fit(x_discovery, y_discovery)

print(f"Accuracy Discovery: {LR.score(x_discovery, y_discovery)}")
print(f"Accuracy Test: {LR.score(x_test, y_test)}")

y_pred_discovery = LR.predict(x_discovery)
y_pred_test = LR.predict(x_test)

from sklearn.metrics import f1_score
f1_discovery = f1_score(y_discovery, y_pred_discovery, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')
print(f"F1 Score Discovery: {f1_discovery}")
print(f"F1 Score Test: {f1_test}")

from sklearn.metrics import roc_auc_score
# 预测概率
y_prob_discovery = LR.predict_proba(x_discovery)[:, 1]
y_prob_test = LR.predict_proba(x_test)[:, 1]

# 计算AUC-ROC分数
roc_auc_discovery = roc_auc_score(y_discovery, y_prob_discovery)
roc_auc_test = roc_auc_score(y_test, y_prob_test)
print(f"AUC-ROC Discovery: {roc_auc_discovery}")
print(f"AUC-ROC Test: {roc_auc_test}")
