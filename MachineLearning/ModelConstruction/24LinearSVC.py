from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, KFold
import optuna

def objective(trial):
    C = trial.suggest_float("C", 1e-5, 1e2)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    # 如果惩罚是 'l1'，则将损失函数固定为 'squared_hinge'
    if penalty == "l1":
        loss = "squared_hinge"
    else:
        loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])

    lsvcHyper = LinearSVC(C=C,
                          penalty=penalty,
                          loss=loss,
                          max_iter=100000)

    # 使用KFold进行10折交叉验证
    kf = KFold(n_splits=10)
    score = cross_val_score(lsvcHyper, x_discovery, y_discovery, cv=kf, scoring='accuracy')
    return score.mean()


# 创建Optuna优化器
study = optuna.create_study(direction='maximize')

# 进行10次优化
for _ in range(10):
    study.optimize(objective, n_trials=100)

print(study.best_params)

best_params = study.best_params

# 根据最佳惩罚类型来设置损失函数
if best_params['penalty'] == 'l1':
    loss = 'squared_hinge'
else:
    loss = best_params['loss']

lsvc = LinearSVC(C=best_params['C'],
                 penalty=best_params['penalty'],
                 loss=loss,
                 max_iter=100000)

lsvc.fit(x_discovery, y_discovery)

print(f"Accuracy Discovery: {lsvc.score(x_discovery, y_discovery)}")
print(f"Accuracy Test: {lsvc.score(x_test, y_test)}")

y_pred_discovery = lsvc.predict(x_discovery)
y_pred_test = lsvc.predict(x_test)

from sklearn.metrics import f1_score
f1_discovery = f1_score(y_discovery, y_pred_discovery, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')
print(f"F1 Score Discovery: {f1_discovery}")
print(f"F1 Score Test: {f1_test}")

from sklearn.metrics import roc_auc_score
# 预测概率
y_scores_discovery = lsvc.decision_function(x_discovery)
y_scores_test = lsvc.decision_function(x_test)

# 计算AUC-ROC分数
roc_auc_discovery = roc_auc_score(y_discovery, y_scores_discovery)
roc_auc_test = roc_auc_score(y_test, y_scores_test)
print(f"AUC-ROC Discovery: {roc_auc_discovery}")
print(f"AUC-ROC Test: {roc_auc_test}")
