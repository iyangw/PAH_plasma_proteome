from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold
import optuna

def objective(trial):
    solver = trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen'])
    shrinkage = None
    if solver in ['lsqr', 'eigen']:
        shrinkage = trial.suggest_float('shrinkage', 0.0, 1.0)  # 缩减范围

    # 定义 LDA 模型
    LDAHyper = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)


    # 使用KFold进行10折交叉验证
    kf = KFold(n_splits=10)
    score = cross_val_score(LDAHyper, x_discovery, y_discovery, cv=kf, scoring='accuracy')
    return score.mean()


# 创建Optuna优化器
study = optuna.create_study(direction='maximize')

# 进行10次优化
for _ in range(10):
    study.optimize(objective, n_trials=100)

print(study.best_params)

best_params = study.best_params

LDA = LinearDiscriminantAnalysis(solver=best_params['solver'], shrinkage=best_params['shrinkage'])

LDA.fit(x_discovery, y_discovery)

print(f"Accuracy Discovery: {LDA.score(x_discovery, y_discovery)}")
print(f"Accuracy Test: {LDA.score(x_test, y_test)}")

y_pred_discovery = LDA.predict(x_discovery)
y_pred_test = LDA.predict(x_test)

from sklearn.metrics import f1_score
f1_discovery = f1_score(y_discovery, y_pred_discovery, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')
print(f"F1 Score Discovery: {f1_discovery}")
print(f"F1 Score Test: {f1_test}")

from sklearn.metrics import roc_auc_score
# 预测概率
y_scores_discovery = LDA.decision_function(x_discovery)
y_scores_test = LDA.decision_function(x_test)

# 计算AUC-ROC分数
roc_auc_discovery = roc_auc_score(y_discovery, y_scores_discovery)
roc_auc_test = roc_auc_score(y_test, y_scores_test)
print(f"AUC-ROC Discovery: {roc_auc_discovery}")
print(f"AUC-ROC Test: {roc_auc_test}")
