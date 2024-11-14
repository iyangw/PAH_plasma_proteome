from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import cross_val_score, KFold
import optuna

def objective(trial):
    n_restarts_optimizer = trial.suggest_int('n_restarts_optimizer', 0, 10)
    optimizer = trial.suggest_categorical('optimizer', ['fmin_l_bfgs_b', None])

    length_scale = trial.suggest_float("length_scale", 1e-2, 1e2)
    noise_level = trial.suggest_float("noise_level", 1e-6, 1e1)
    kernel = ConstantKernel(noise_level) * RBF(length_scale=length_scale, length_scale_bounds=(1e-5, 1e308))


    GPCHyper = GaussianProcessClassifier(kernel=kernel,
                                         n_restarts_optimizer=n_restarts_optimizer,
                                         optimizer=optimizer,
                                         max_iter_predict=1000)


    # 使用KFold进行10折交叉验证
    kf = KFold(n_splits=10)
    score = cross_val_score(GPCHyper, x_discovery, y_discovery, cv=kf, scoring='accuracy')
    return score.mean()


# 创建Optuna优化器
study = optuna.create_study(direction='maximize')

# 进行10次优化
for _ in range(10):
    study.optimize(objective, n_trials=100)

print(study.best_params)

best_params = study.best_params

final_kernel = ConstantKernel(best_params['noise_level']) * RBF(length_scale=best_params['length_scale'], length_scale_bounds=(1e-5, 1e308))
GPC = GaussianProcessClassifier(kernel=final_kernel,
                                n_restarts_optimizer=best_params['n_restarts_optimizer'],
                                optimizer=best_params['optimizer'])

GPC.fit(x_discovery, y_discovery)

print(f"Accuracy Discovery: {GPC.score(x_discovery, y_discovery)}")
print(f"Accuracy Test: {GPC.score(x_test, y_test)}")

y_pred_discovery = GPC.predict(x_discovery)
y_pred_test = GPC.predict(x_test)

from sklearn.metrics import f1_score
f1_discovery = f1_score(y_discovery, y_pred_discovery, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')
print(f"F1 Score Discovery: {f1_discovery}")
print(f"F1 Score Test: {f1_test}")

from sklearn.metrics import roc_auc_score
# 预测概率
y_probs_discovery = GPC.predict_proba(x_discovery)[:, 1]
y_probs_test = GPC.predict_proba(x_test)[:, 1]

# 计算AUC-ROC分数
roc_auc_discovery = roc_auc_score(y_discovery, y_probs_discovery)
roc_auc_test = roc_auc_score(y_test, y_probs_test)
print(f"AUC-ROC Discovery: {roc_auc_discovery}")
print(f"AUC-ROC Test: {roc_auc_test}")
