from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.svm import SVC

best_params = {'C': 31.854231355456225, 'kernel': 'linear', 'degree': 5, 'gamma': 'auto', 'coef0': 0.15697852120652345}


svc = SVC(C=best_params['C'],
          kernel=best_params['kernel'],
          degree=best_params['degree'],
          gamma=best_params['gamma'],
          coef0=best_params['coef0'],
          probability=True)

svc.fit(x_discovery, y_discovery)

from BorutaShap import BorutaShap
# no model selected default is Random Forest, if classification is True it is a Classification problem
Feature_Selector = BorutaShap(model=svc,
                              importance_measure='shap',
                              classification=True)


Feature_Selector.fit(X=x_discovery, y=y_discovery,
                     n_trials=100,
                     sample=False,
                     normalize=True)
