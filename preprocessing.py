from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.feature_selection import SelectPercentile, RFECV
from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import make_column_transformer
from sklearn.svm import SVR
import hyperopt.hp as hp
import numpy as np
import pandas as pd

feature_preprocessing = hp.choice('feature_preprocessor', [
                                  {'PCA' : PCA(),
                                   'parameters' :
                                   {'n_components' : hp.uniform('n_components_PCA', 0.3, 0.8)}
                                   },
                                   {'KernelPCA' : KernelPCA(),
                                    'parameters' :
                                    {'n_components' : hp.choice('n_components_KernelPCA', range(3, 13))}
                                   },
                                   {'FastICA' : FastICA(),
                                    'parameters' :
                                    {'n_components' : hp.choice('n_components_FastICA', range(3, 13))}
                                   },
                                   {'SelectPercentile' : SelectPercentile(),
                                    'parameters' : {}
                                   },
                                   {'RFECV' : RFECV(SVR(kernel='linear')),
                                    'parameters' : {}
                                   },
                                   {'FeatureAgglomeration' : FeatureAgglomeration(),
                                    'parameters' :
                                    {'n_clusters' : hp.choice('n_clusters_FeatureAgglomeration', range(2, 6))}
                                   }
                                   ])

# TODO: Scale, normalize what to do?
def preprocess_data_train():
    url = 'http://www.it.uu.se/edu/course/homepage/sml/project/training_data.csv'
    data = pd.read_csv(url)
    Y_data = data['label']
    X_data = data.drop(['label'], axis=1)

    X_data['duration'] = MinMaxScaler().fit_transform(np.array(X_data['duration']).reshape(-1, 1))
#    X_data = OneHotEncoder(categories='auto').fit_transform(X_data)
    X_data = MinMaxScaler().fit_transform(X_data)
#    Normalizer().fit_transform(X_data)

    return X_data, Y_data

def preprocess_data_test():
    url = 'http://www.it.uu.se/edu/course/homepage/sml/project/songs_to_classify.csv'
    X_data = pd.read_csv(url)

    X_data['duration'] = MinMaxScaler().fit_transform(np.array(X_data['duration']).reshape(-1, 1))
#    OneHotEncoder(categories='auto').fit_transform(X_data)
    X_data = MinMaxScaler().fit_transform(X_data)
#    Normalizer().fit_transform(X_data)

    return X_data
