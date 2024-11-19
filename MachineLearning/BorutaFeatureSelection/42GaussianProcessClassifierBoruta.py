from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.drop(columns=['Number', 'Condition'])
y_discovery = discovery.loc[:, 'Condition']

x_test = test.drop(columns=['Number', 'Condition'])
y_test = test.loc[:, 'Condition']

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

best_params = {'n_restarts_optimizer': 6, 'optimizer': 'fmin_l_bfgs_b', 'length_scale': 88.41399263256385, 'noise_level': 8.17156436908568}


final_kernel = ConstantKernel(best_params['noise_level']) * RBF(length_scale=best_params['length_scale'], length_scale_bounds=(1e-5, 1e308))
GPC = GaussianProcessClassifier(kernel=final_kernel,
                                n_restarts_optimizer=best_params['n_restarts_optimizer'],
                                optimizer=best_params['optimizer'])

GPC.fit(x_discovery, y_discovery)


from BorutaShap import BorutaShap
# no model selected default is Random Forest, if classification is True it is a Classification problem
Feature_Selector = BorutaShap(model=GPC,
                              importance_measure='shap',
                              classification=True)


Feature_Selector.fit(X=x_discovery, y=y_discovery,
                     n_trials=100,
                     sample=False,
                     normalize=True)
