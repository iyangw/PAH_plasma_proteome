from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.ensemble import ExtraTreesClassifier

best_params = {'n_estimators': 285, 'max_depth': 21, 'min_samples_split': 7, 'min_samples_leaf': 6, 'max_features': 278, 'max_leaf_nodes': 41, 'min_impurity_decrease': 0.10953951841280787, 'ccp_alpha': 0.0756759705112319}


ETs = ExtraTreesClassifier(
           n_estimators=best_params['n_estimators'],
           max_depth=best_params['max_depth'],
           min_samples_split=best_params['min_samples_split'],
           min_samples_leaf=best_params['min_samples_leaf'],
           max_leaf_nodes=best_params['max_leaf_nodes'],
           max_features=best_params['max_features'],
           min_impurity_decrease=best_params['min_impurity_decrease'],
           ccp_alpha=best_params['ccp_alpha'])


from BorutaShap import BorutaShap
# no model selected default is Random Forest, if classification is True it is a Classification problem
Feature_Selector = BorutaShap(model=ETs,
                              importance_measure='shap',
                              classification=True)


Feature_Selector.fit(X=x_discovery, y=y_discovery,
                     n_trials=100,
                     sample=False,
                     normalize=True)
