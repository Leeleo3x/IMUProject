import numpy as np
import sklearn.svm as svm
import sklearn.datasets as datasets
from sklearn.metrics import mean_squared_error
import training_data as td
from multiprocessing import Pool, Lock
import multiprocessing

class SVRGridSearch:
    """
    Grid search for SVR
    """
    def __init__(self, search_dict, num_threads=6, verbose=True):
        """
        Constructor
        :param search_dict: dictionary of searching parameters, currently support C and epsilon for SVR
        :param num_threads: number of processes to use
        """
        self.search_dict_ = search_dict
        self.num_threads_ = num_threads
        self.best_score_ = np.inf
        self.best_param_ = {}
        self.verbose_ = verbose
        self.lock_ = Lock()
        self.training_set_ = None
        self.validation_set_ = None

    def set_search_dict(self, new_search_dict):
        self.search_dict_ = new_search_dict

    def worker(self, param_dict):
        """
        Fit SVR based on one specific parameter
        :param param_dict: dictionary of parameter
        :return: None
        """
        c, e = param_dict['c'], param_dict['e']
        regressor = svm.SVR(C=c, epsilon=e)
        regressor.fit(self.training_set_[:, :-1], self.training_set_[:, -1])
        #score = regressor.score(self.validation_set_[:, :-1], self.validation_set_[:, -1])
        score = mean_squared_error(regressor.predict(self.validation_set_[:, :-1]), self.validation_set_[:, -1])
        print('param: C={}, e={}, score={}'.format(c, e, score))
        if score < self.best_score_:
            self.best_score_ = score
            self.best_param_['c'] = c
            self.best_param_['e'] = e

    def run(self, data, hold_off_ratio=0.3):
        """
        Run grid search
        :param train: training set, Nxd array
        :param target: ground truth value, Nx1 array
        :param hold_off_ratio: ratio of hold_off validation set
        :return: dictionary of optimal parameter, optimal r2 score
        """
        assert 'c' in self.search_dict_, "Missing parameter 'c'"
        assert 'e' in self.search_dict_, "Missing parameter 'e'"
        self.training_set_, self.validation_set_ = td.split_data(data, hold_off_ratio)
        # with Pool(self.num_threads_) as p:
        #     p.map(self.worker, [{'c': c, 'e': e}
        #                         for c in self.search_dict_['c']
        #                         for e in self.search_dict_['e']])

        [self.worker({'c': c, 'e': e})
         for c in self.search_dict_['c']
         for e in self.search_dict_['e']]
        return self.best_param_, self.best_score_

    def reset(self):
        self.best_score_ = -np.inf
        self.best_param_ = {}
        self.training_set_ = None
        self.validation_set_ = None

if __name__ == '__main__':
    boston = datasets.load_boston()
    training_set = np.concatenate([boston.data, boston.target[:, None]], axis=1)
    search_dict = {'c': [0.1, 0.5, 1.0, 5.0, 1.0, 20.0],
                   'e': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]}
    grid_searcher = SVRGridSearch(search_dict)
    best_param, best_score = grid_searcher.run(training_set)

    print('Best param:\n')
    print(best_param)
    print('Best score: ', best_score)
