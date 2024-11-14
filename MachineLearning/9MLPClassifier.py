from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
import optuna

def objective(trial):
    hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes', 1, 200)  # 选择隐藏层大小
    activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1)  # 正则化强度
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-5, 1e-1)  # 初始学习率
    learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])  # 学习率策略
    power_t = trial.suggest_float('power_t', 0.1, 1.0)  # 用于反比例缩放学习率

    MLPHyper = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,),
                             activation=activation,
                             alpha=alpha,
                             learning_rate=learning_rate,
                             learning_rate_init=learning_rate_init,
                             power_t=power_t,
                             max_iter=2000)

    # 使用KFold进行10折交叉验证
    kf = KFold(n_splits=10)
    score = cross_val_score(MLPHyper, x_discovery, y_discovery, cv=kf, scoring='accuracy')
    return score.mean()


# 创建Optuna优化器
study = optuna.create_study(direction='maximize')

# 进行10次优化
for _ in range(10):
    study.optimize(objective, n_trials=100)

print(study.best_params)

best_params = study.best_params

MLP = MLPClassifier(hidden_layer_sizes=(best_params['hidden_layer_sizes'],),
                    activation=best_params['activation'],
                    alpha=best_params['alpha'],
                    learning_rate=best_params['learning_rate'],
                    learning_rate_init=best_params['learning_rate_init'],
                    power_t=best_params['power_t'],
                    max_iter=2000)

MLP.fit(x_discovery, y_discovery)

print(f"Accuracy Discovery: {MLP.score(x_discovery, y_discovery)}")
print(f"Accuracy Test: {MLP.score(x_test, y_test)}")

y_pred_discovery = MLP.predict(x_discovery)
y_pred_test = MLP.predict(x_test)

from sklearn.metrics import f1_score
f1_discovery = f1_score(y_discovery, y_pred_discovery, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')
print(f"F1 Score Discovery: {f1_discovery}")
print(f"F1 Score Test: {f1_test}")

from sklearn.metrics import roc_auc_score
# 预测概率
y_prob_discovery = MLP.predict_proba(x_discovery)[:, 1]
y_prob_test = MLP.predict_proba(x_test)[:, 1]

# 计算AUC-ROC分数
roc_auc_discovery = roc_auc_score(y_discovery, y_prob_discovery)
roc_auc_test = roc_auc_score(y_test, y_prob_test)
print(f"AUC-ROC Discovery: {roc_auc_discovery}")
print(f"AUC-ROC Test: {roc_auc_test}")
