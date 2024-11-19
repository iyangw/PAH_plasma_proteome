from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.neighbors import KNeighborsClassifier


best_params = {'n_neighbors': 4, 'weights': 'distance', 'metric': 'euclidean'}


KNeigh = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'],
                              weights=best_params['weights'],
                              metric=best_params['metric'])

KNeigh.fit(x_discovery, y_discovery)


from BorutaShap import BorutaShap
# no model selected default is Random Forest, if classification is True it is a Classification problem
Feature_Selector = BorutaShap(model=KNeigh,
                              importance_measure='shap',
                              classification=True)


Feature_Selector.fit(X=x_discovery, y=y_discovery,
                     n_trials=100,
                     sample=False,
                     normalize=True)
