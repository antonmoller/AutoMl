# Silence failed to converge warnings from sklearn.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials

import hyperparameters
import preprocessing

X_train, Y_train = preprocessing.preprocess_data_train()
test = preprocessing.preprocess_data_test()
MAX_EVALS = 10
NO_PIPELINES = 50
FOLDS = 10
SCORING = 'accuracy'
BEST_PIPELINES = {}

space = hp.choice('classifier', [
                  {'LogisticRegression' : LogisticRegression(class_weight='balanced'),
                   'parameters' : hyperparameters.LogisticRegression,
                   'preprocessing' : preprocessing.feature_preprocessing
                  },
                  {'Ridge' : RidgeClassifier(class_weight='balanced'),
                   'parameters' : hyperparameters.Ridge,
                   'preprocessing' : preprocessing.feature_preprocessing
                  },
                  {'LDA' : LinearDiscriminantAnalysis(),
                   'parameters' : hyperparameters.LDA,
                   'preprocessing' : preprocessing.feature_preprocessing
                  },
                  {'QDA' : QuadraticDiscriminantAnalysis(),
                   'parameters' : hyperparameters.QDA,
                   'preprocessing' : preprocessing.feature_preprocessing
                  },
                  {'KNN' : KNeighborsClassifier(),
                   'parameters' : hyperparameters.KNN,
                   'preprocessing' : preprocessing.feature_preprocessing
                  },
                  {'DecisionTree' : DecisionTreeClassifier(class_weight='balanced'),
                   'parameters' : hyperparameters.DecisionTree,
                   'preprocessing' : preprocessing.feature_preprocessing
                  },
                  {'ExtraTree' : ExtraTreeClassifier(class_weight='balanced'),
                   'parameters' : hyperparameters.ExtraTree,
                   'preprocessing' : preprocessing.feature_preprocessing
                  },
                  {'AdaBoost' : AdaBoostClassifier().fit(X_train, Y_train), # WHY IS FITTING NEEDED???
                   'parameters' : hyperparameters.AdaBoost,
                   'preprocessing' : preprocessing.feature_preprocessing
                  },
                  {'Gradientboosting' : GradientBoostingClassifier().fit(X_train, Y_train),
                   'parameters' : hyperparameters.GradientBoosting,
                   'preprocessing' : preprocessing.feature_preprocessing
                 },
                  {'Bagging' : BaggingClassifier().fit(X_train, Y_train),
                   'parameters' : hyperparameters.Bagging,
                   'preprocessing' : preprocessing.feature_preprocessing
                  },
                  {'RandomForest' : RandomForestClassifier().fit(X_train, Y_train),
                   'parameters' : hyperparameters.RandomForest,
                   'preprocessing' : preprocessing.feature_preprocessing
                  }
                ])

def save_pipeline(pipeline, score):
    if len(BEST_PIPELINES) != 0:
        lowest_score = min(BEST_PIPELINES, key=float)

    if len(BEST_PIPELINES) < NO_PIPELINES:
        pipeline.fit(X_train, Y_train)
        BEST_PIPELINES[score] = pipeline
    elif score > lowest_score:
        pipeline.fit(X_train, Y_train)
        BEST_PIPELINES[score] = pipeline
        del BEST_PIPELINES[lowest_score]

def objective(pipeline):
    name = list(pipeline.keys())[0]
    pipeline = list(pipeline.values())
    model = pipeline[0]
    params_model = pipeline[1]
    model.set_params(**params_model)

    pre = list(pipeline[2].values())[0]
    params_pre = list(pipeline[2].values())[1]
    pre.set_params(**params_pre)
    pre.fit_transform(X_train, Y_train)

    pipeline_classification = make_pipeline(pre, model)

    score = cross_val_score(pipeline_classification, X_train, Y_train, scoring=SCORING, cv=FOLDS)
    save_pipeline(pipeline_classification, score.mean())

    print(name + ": " + str(score.mean())[:4] + " (+/-" + str(score.std())[:5] + ")")

    return {'loss' : 1 - score.mean(), 'status' : STATUS_OK}

def create_ensemble(ens_size):
    pipelines = BEST_PIPELINES

    for i in range(0, ens_size):


trials = Trials()
best = fmin(objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials,
            return_argmin=True)
print(best)
print(1 - trials.best_trial['result']['loss'])

