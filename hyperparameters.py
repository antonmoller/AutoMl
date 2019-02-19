from hyperopt import hp

LogisticRegression = {'penalty' : hp.choice('penalty_LogReg', ('l1', 'l2')),
                      'C' : hp.uniform('C', 0, 2),
                      'fit_intercept' : hp.choice('fit_intercept_LogReg', (True, False))
                     }

Ridge = {'alpha' : hp.uniform('alpha_Ridge', 0.01, 0.3),
         'fit_intercept' : hp.choice('fit_intercept_Ridge', (True, False)),
         'normalize' : hp.choice('normalize_Ridge', (True, False)),
        }

LDA = {'n_components' : hp.choice('n_components_LDA', range(1, 20))
      }

QDA = {'reg_param' : hp.uniform('reg_parm_QDA', 0, 0.5)}

KNN = {'n_neighbors' : hp.choice('n_neigbors_KNN', range(1, 300)),
       'algorithm' : hp.choice('algorithm_KNN', ('auto', 'ball_tree', 'kd_tree', 'brute')),
       'leaf_size' : hp.choice('leaf_size_KNN', range(10, 100)),
       'metric' : hp.choice('metric_KNN', ('cityblock', 'euclidean', 'l1', 'l2', 'manhattan'))
      }

DecisionTree = {'splitter' : hp.choice('split_DecisionTree', ('best', 'random')),
                'max_depth' : hp.choice('max_depth_DecisionTree', range(1, 20)),
                'min_samples_split' : hp.uniform('min_samples_split_DecisionTree', 0.1, 0.6),
                'max_features' : hp.choice('max_features_DecisionTree', ('auto', 'sqrt', 'log2', None)),
                'max_leaf_nodes' : hp.choice('max_leaf_nodes_DecisionTree', range(2, 100))
               }

ExtraTree =  {'splitter' : hp.choice('split_ExtraTrees', ('best', 'random')),
              'max_depth' : hp.choice('max_depth_ExtraTrees', range(1, 20)),
              'min_samples_split' : hp.uniform('min_samples_split_ExtraTrees', 0.1, 0.6),
              'max_features' : hp.choice('max_features_ExtraTrees', ('auto', 'sqrt', 'log2', None)),
              'max_leaf_nodes' : hp.choice('max_leaf_nodes_ExtraTrees', range(2, 100))
               }

AdaBoost = {'n_estimators' : hp.choice('n_estimators_Ada', range(50, 500)),
            'learning_rate' : hp.uniform('learning_rate_Ada', 0.01, 0.5),
            'algorithm' : hp.choice('algorithm_Ada', ('SAMME', 'SAMME.R'))
           }

Bagging = {'n_estimators' : hp.choice('n_estimators_Bagging', range(10, 300)),
           'max_samples' : hp.uniform('max_samples_Bagging', 0.1, 0.3)
          }

GradientBoosting = {'learning_rate' : hp.uniform('learning_rate_GradientBoosting', 0.01, 0.5),
                    'n_estimators' : hp.choice('n_estimators_GradientBoosting', range(50, 200)),
                    'min_samples_split' : hp.uniform('min_samples_split_GradientBoosting', 0.1, 0.4),
                    'min_samples_leaf' : hp.uniform('min_samples_leaf_GradientBoosting', 0.1, 0.4),
                    'min_weight_fraction_leaf' : hp.uniform('min_weight_fraction_leaf_GradientBoosting', 0.1, 0.5),
                    'max_features' : hp.uniform('max_features_GradientBoosting', 0.1, 0.8),
                    'max_depth' : hp.choice('max_depth_Boosting', range(1, 10))
                   }

RandomForest = {'n_estimators' : hp.choice('n_estimators_RandomForest', range(10, 500)),
                'max_depth' : hp.choice('max_depth_RandomForest', range(10, 40)),
                'min_samples_split' : hp.uniform('min_samples_split_RandomForest', 0.1, 0.4),
                'min_samples_leaf' : hp.uniform('min_samples_leaf_RandomForest', 0.1, 0.4),
                'max_features' : hp.uniform('max_features_RandomForest', 0.1, 0.8),
                'bootstrap' : hp.choice('bootstrap_RandomForest', (True, False)),
                'class_weight' : hp.choice('class_weight_RandomForest', ('balanced', 'balanced_subsample'))
               }
