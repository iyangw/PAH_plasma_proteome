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

best_params = {'C': 0.021723031822130923, 'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.00216395361426458}

LR = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], solver=best_params['solver'], l1_ratio=best_params['l1_ratio'], max_iter=10000)


from BorutaShap import BorutaShap
# no model selected default is Random Forest, if classification is True it is a Classification problem
Feature_Selector = BorutaShap(model=LR,
                              importance_measure='shap',
                              classification=True)


Feature_Selector.fit(X=x_discovery, y=y_discovery,
                     n_trials=100,
                     normalize=True,
                     sample=False)

