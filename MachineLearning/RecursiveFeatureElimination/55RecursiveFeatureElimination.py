from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import shap
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class RFEshap_test:
    '''
    RFE using CV and SHAPS
    '''

    def __init__(self, reserved_features, dropped_features, scores):
        self.reserved_features = reserved_features
        self.dropped_features = dropped_features
        self.scores = scores


def RFECV_shap(
        data_dict: dict,
        params: dict = None,
        random_seed: int = None,
        nfolds: int = 10,
        min_features: int = 2
):
    # initialize stratified fold split
    folds = KFold(
        n_splits=nfolds,
        shuffle=True,
        random_state=random_seed
    )

    # initialize list to document metrics across numbers of features
    metric_dict = {}
    reserved_features_dict = {}
    dropped_features_dict = {}

    features = list(data_dict['X'].columns)
    drop_feature = None

    for i, vars in enumerate(features):

        # set features list
        if i == 0:
            features = features
            reserved_features_dict.update({f'{len(features)}': features})
        else:
            features = [x for x in features if x != drop_feature]
            reserved_features_dict.update({f'{len(features)}': features})

        # print progress
        print(f'{len(features)}->{len(features) - 1}')

        # update data dict with features for this round
        data_dict.update({'X': data_dict['X'][features]})

        # initialize list to document metrics across folds
        scores = []
        drop_features = []

        # split train data into folds, fit model within each fold
        for i, (train_idxs, val_idxs) in enumerate(folds.split(data_dict['X'], data_dict['y'])):
            eval_data = {
                'X_val': data_dict['X'].iloc[val_idxs],
                'y_val': data_dict['y'].iloc[val_idxs],
                'X_train': data_dict['X'].iloc[train_idxs],
                'y_train': data_dict['y'].iloc[train_idxs]
            }

            # initialize model
            model = KNeighborsClassifier(**params)

            model.fit(
                X=eval_data['X_train'],
                y=eval_data['y_train']
            )

            # get predicted values
            preds = model.predict(eval_data['X_val'])

            # get accuracy in the fold
            accuracy = accuracy_score(eval_data['y_val'], preds)

            # append to list of R squared across folds
            scores.append(accuracy)

            # get shap values
            explainer = shap.KernelExplainer(model.predict_proba, eval_data['X_train'])
            shap_values = explainer.shap_values(eval_data['X_train'])[:, :, 1]

            # aggregate SHAP values for each feature
            feature_importance = np.abs(shap_values).mean(axis=0)

            # sort features based on importance scores
            sorted_features = pd.Series(
                feature_importance,
                index=eval_data['X_train'].columns
            ).sort_values(ascending=False)

            # select the worst feature
            drop_features.append(sorted_features.index[-1])

        # get average R squared across folds
        avg_score = np.mean(scores)
        # update dict of metrics
        metric_dict.update({f'{len(features)}': avg_score})

        if len(features) <= min_features:
            break

        # get most common feature ranked last
        element_counts = Counter(drop_features)
        drop_feature = max(element_counts, key=element_counts.get)
        dropped_features_dict.update({f'{len(features)}->{len(features)-1}': drop_feature})

    # make class object
    results = RFEshap_test(
        reserved_features_dict,
        dropped_features_dict,
        metric_dict
    )
    print(reserved_features_dict)
    print(dropped_features_dict)
    print(metric_dict)
    return results

from sklearn.model_selection import train_test_split

# Load the CSV into a DataFrame
data = pd.read_csv('MachineLearningSourceData.csv')

discovery, test = train_test_split(data, test_size=0.2, random_state=3220822)

x_discovery = discovery.loc[:, ['GDI2', 'C7', 'FCGR3A', 'IGKV1-12', 'TTR', 'CRTAC1', 'ICAM3', 'HSPA5', 'PNP', 'RNH1', 'DEFA1', 'ABCA1', 'CRISP3']]
y_discovery = discovery.loc[:, 'Condition']

x_test = test.loc[:, ['GDI2', 'C7', 'FCGR3A', 'IGKV1-12', 'TTR', 'CRTAC1', 'ICAM3', 'HSPA5', 'PNP', 'RNH1', 'DEFA1', 'ABCA1', 'CRISP3']]
y_test = test.loc[:, 'Condition']


post_boruta_dict = {
    'X': x_discovery,
    'y': y_discovery
}

best_params = {'n_neighbors': 18, 'weights': 'distance', 'metric': 'manhattan'}

rfecv_shap = RFECV_shap(
    data_dict=post_boruta_dict,
    params=best_params,
    min_features=2,
    nfolds=10
)

