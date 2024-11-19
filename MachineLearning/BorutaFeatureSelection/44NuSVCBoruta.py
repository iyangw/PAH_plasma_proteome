from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.svm import NuSVC

best_params = {'nu': 0.15205417150191694, 'kernel': 'linear', 'degree': 4, 'gamma': 'scale', 'coef0': 0.8262157776523615}

nusvc = NuSVC(nu=best_params['nu'],
              kernel=best_params['kernel'],
              degree=best_params['degree'],
              gamma=best_params['gamma'],
              probability=True)

nusvc.fit(x_discovery, y_discovery)

from BorutaShap import BorutaShap
# no model selected default is Random Forest, if classification is True it is a Classification problem
Feature_Selector = BorutaShap(model=nusvc,
                              importance_measure='shap',
                              classification=True)


Feature_Selector.fit(X=x_discovery, y=y_discovery,
                     n_trials=100,
                     sample=False,
                     normalize=True)
