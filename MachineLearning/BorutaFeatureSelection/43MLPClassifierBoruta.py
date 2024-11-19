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

best_params = {'hidden_layer_sizes': 76, 'activation': 'logistic', 'alpha': 0.08877319523674633, 'learning_rate_init': 0.013094435205084173, 'learning_rate': 'constant', 'power_t': 0.20565620128728787}


MLP = MLPClassifier(hidden_layer_sizes=(best_params['hidden_layer_sizes'],),
                    activation=best_params['activation'],
                    alpha=best_params['alpha'],
                    learning_rate=best_params['learning_rate'],
                    learning_rate_init=best_params['learning_rate_init'],
                    power_t=best_params['power_t'],
                    max_iter=2000)

MLP.fit(x_discovery, y_discovery)


from BorutaShap import BorutaShap
# no model selected default is Random Forest, if classification is True it is a Classification problem
Feature_Selector = BorutaShap(model=MLP,
                              importance_measure='shap',
                              classification=True)


Feature_Selector.fit(X=x_discovery, y=y_discovery,
                     n_trials=100,
                     sample=False,
                     normalize=True)