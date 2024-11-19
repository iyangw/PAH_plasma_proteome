from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.ensemble import HistGradientBoostingClassifier

best_params = {'learning_rate': 0.37200697751151546, 'max_iter': 389, 'max_leaf_nodes': 35, 'max_depth': 4, 'min_samples_leaf': 11, 'l2_regularization': 3.564500823776875, 'max_features': 0.14307701640758047}


HGBC = HistGradientBoostingClassifier(learning_rate=best_params['learning_rate'],
                                      max_iter=best_params['max_iter'],
                                      max_leaf_nodes=best_params['max_leaf_nodes'],
                                      max_depth=best_params['max_depth'],
                                      min_samples_leaf=best_params['min_samples_leaf'],
                                      l2_regularization=best_params['l2_regularization'],
                                      max_features=best_params['max_features'])

HGBC.fit(x_discovery, y_discovery)


from BorutaShap import BorutaShap
# no model selected default is Random Forest, if classification is True it is a Classification problem
Feature_Selector = BorutaShap(model=HGBC,
                              importance_measure='shap',
                              classification=True)


Feature_Selector.fit(X=x_discovery, y=y_discovery,
                     n_trials=100,
                     sample=False,
                     normalize=True)

