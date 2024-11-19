from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.linear_model import Perceptron

best_params = {'penalty': 'elasticnet', 'alpha': 0.001766683226587159, 'eta0': 8.878333422133545, 'l1_ratio': 0.00397014841645112}


perceptron = Perceptron(penalty=best_params['penalty'],
                        alpha=best_params['alpha'],
                        l1_ratio=best_params['l1_ratio'],
                        eta0=best_params['eta0'],
                        max_iter=1000)

perceptron.fit(x_discovery, y_discovery)


from BorutaShap import BorutaShap
# no model selected default is Random Forest, if classification is True it is a Classification problem
Feature_Selector = BorutaShap(model=perceptron,
                              importance_measure='shap',
                              classification=True)


Feature_Selector.fit(X=x_discovery, y=y_discovery,
                     n_trials=100,
                     sample=False,
                     normalize=True)
