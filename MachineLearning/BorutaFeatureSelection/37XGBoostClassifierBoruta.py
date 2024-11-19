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

best_params = {'booster': 'gblinear', 'learning_rate': 0.1732563228205146, 'n_estimators': 294, 'reg_alpha': 0.012887626962138854, 'reg_lambda': 1.628444710663937}

XGB = XGBClassifier(**best_params)

XGB.fit(x_discovery, y_discovery)


from BorutaShap import BorutaShap
# no model selected default is Random Forest, if classification is True it is a Classification problem
Feature_Selector = BorutaShap(model=XGB,
                              importance_measure='shap',
                              classification=True)


Feature_Selector.fit(X=x_discovery, y=y_discovery,
                     n_trials=100,
                     sample=False,
                     normalize=True)

