from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.ensemble import BaggingClassifier

best_params = {'n_estimators': 39, 'max_samples': 0.23055317373795903, 'max_features': 0.534132687244522}


BC = BaggingClassifier(n_estimators=best_params['n_estimators'],
                       max_samples=best_params['max_samples'],
                       max_features=best_params['max_features'])

BC.fit(x_discovery, y_discovery)


from BorutaShap import BorutaShap
# no model selected default is Random Forest, if classification is True it is a Classification problem
Feature_Selector = BorutaShap(model=BC,
                              importance_measure='shap',
                              classification=True)

Feature_Selector.fit(X=x_discovery, y=y_discovery,
                     n_trials=100,
                     sample=False,
                     normalize=True)
