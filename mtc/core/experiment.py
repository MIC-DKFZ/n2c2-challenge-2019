"""
.. module:: experiment
   :synopsis: Holding all experiment classes!
.. moduleauthor:: Klaus Kades
"""

import os
import time
from typing import List, Dict, Union
from abc import ABC, abstractmethod
from datetime import datetime
import json
from deprecated import deprecated
from sklearn.preprocessing import StandardScaler


import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mtc.core.sentence import Sentence
from mtc.core.preprocessor import Preprocessor
from mtc.core.evaluator import Evaluator
from mtc.core.similarity_measures import SimilarityMeasures
from mtc.core.embeddings import DocumentEmbeddings
from mtc.core.embeddings import TokenEmbeddings
from mtc.settings import NLP_EXPERIMENT_PATH
from mtc.helpers.decorators import timeit
from mtc.helpers.file_management import load_sts_data


class Measuring:

    @staticmethod
    def preprocess_sentences(preprocessor, raw_texts: List[str]) -> List[Sentence]:
        preprocessed_texts = preprocessor.preprocess(raw_texts)
        return [Sentence(sen) for sen in preprocessed_texts]

    def __init__(self, measures: Union[SimilarityMeasures, List[SimilarityMeasures]], pickle_folder):

        self.measures = measures
        self.pickle_folder = pickle_folder

        self.X = []

        self.sentences_dict = dict()
        self.time_log = []

        if type(self.measures) is SimilarityMeasures:
            self.measures = [self.measures]

    def __call__(self):
        return self.X, self.sentences_dict['y'], np.array(self.sentences_dict['raw_sentences_a']), \
               np.array(self.sentences_dict['raw_sentences_b'])

    @timeit
    def create_features(self):
        self.X = np.array([], dtype=np.int64).reshape(len(self.sentences_dict['raw_sentences_a']), 0)
        if self.measures is not None:
            for measure in self.measures:
                sentences_a, sentences_b = self._get_prerpocessed_sentences(measure.pa_preprocessor)
                measure_values = []
                for sentence_a in sentences_a:
                    feature_to_add = sentence_a.get_sentence_property(measure.key_name)
                    if type(feature_to_add) is not list:
                        feature_to_add = [feature_to_add]
                    measure_values.append(feature_to_add)
                    #measure_values.append(sentence_a.get_sentence_property(measure.key_name))
                self.X = np.append(self.X, np.array(measure_values), axis=1)

        if self.X.size != 0:
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)


    @timeit
    def save_feature_matrix(self):
        pickle.dump(np.nan_to_num(self.X), open(os.path.join(self.pickle_folder, f'feature_matrix.p'), 'wb'))

    @timeit
    def load_feature_matrix(self):
        self.X = pickle.load(open(os.path.join(self.pickle_folder, f'feature_matrix.p'), 'rb'))

    @timeit
    def save_sentences_objects(self):
        pickle.dump(self.sentences_dict, open(os.path.join(self.pickle_folder, f'sentences_dict.p'), 'wb'))

    @timeit
    def load_sentences_objects(self):
        self.sentences_dict = pickle.load(open(os.path.join(self.pickle_folder, f'sentences_dict.p'), 'rb'))

    @timeit
    def set_sts_data_dict(self, sts_data_path):

        try:
            self.load_sentences_objects()
        except FileNotFoundError:
            print('No file was located')
            sts_data = load_sts_data(sts_data_path)
            if 'similarity_score' in sts_data:
                self.sentences_dict.update({
                    'raw_sentences_a': sts_data['raw_sentences_a'],
                    'raw_sentences_b': sts_data['raw_sentences_b'],
                    'y': np.array(sts_data['similarity_score'])
                })
            else:
                self.sentences_dict.update({
                    'raw_sentences_a': sts_data['raw_sentences_a'],
                    'raw_sentences_b': sts_data['raw_sentences_b'],
                    'y': np.arange(0, len(sts_data['raw_sentences_a']))
                })
            self.save_sentences_objects()

    @timeit
    def _get_prerpocessed_sentences(self, pa_preprocessor):
        name = str(pa_preprocessor)
        if name not in self.sentences_dict.keys():
            preprocessor = Preprocessor(*pa_preprocessor['args'], **pa_preprocessor['kwargs'])
            sentences_combined = self.preprocess_sentences(preprocessor, self.sentences_dict['raw_sentences_a'] +
                                                           self.sentences_dict['raw_sentences_b'])
            sentences_a = sentences_combined[:len(sentences_combined) // 2]
            sentences_b = sentences_combined[len(sentences_combined) // 2:]
            # sentences_a = self.preprocess_sentences(preprocessor, self.raw_sentences_a)
            # sentences_b = self.preprocess_sentences(preprocessor, self.raw_sentences_b)
            self.sentences_dict.update({
                name: {'sentences_a': sentences_a, 'sentences_b': sentences_b}
            })
        return self.sentences_dict[name]['sentences_a'], self.sentences_dict[name]['sentences_b']

    @timeit
    def measure(self, train_or_test):
        if self.measures is not None:
            for measure in self.measures:
                sentences_a, sentences_b = self._get_prerpocessed_sentences(measure.pa_preprocessor)
                measure.measure(sentences_a, sentences_b, train_or_test)

    def plot_correlation_matrix(self):
        import matplotlib.pyplot as plt
        import matplotlib
        key_names = [measure.key_name for measure in self.measures]

        for idx, key_name in enumerate(key_names):
            print(idx+2, key_name)

        features = np.concatenate((np.array([self.sentences_dict['y']]).reshape(-1, 1), self.X), axis=1)
        df = pd.DataFrame(features)
        #df.columns = ['ground truth'] + key_names
        plt.matshow(df.corr(method='pearson').abs())
        plt.colorbar()
        plt.show()
        plt.savefig('correlation.png')


    @timeit
    def save_measuring_config(self):

        # experiment_params
        experiment_params = dict()
        if self.measures is not None:
            experiment_params['measures'] = [measure.key_name for measure in self.measures]

        experiment_params['timings'] = self.time_log
        with open(os.path.join(self.pickle_folder, f'experiment_parameters.json'), 'w+') as fp:
            json.dump(experiment_params, fp)


class Training:

    def __init__(self,  estimator):

        self.estimator = estimator

    def __call__(self, X_train, y_train):
        self.estimator.fit(X_train, y_train)

    def save_training_properties(self, experiment_path,  pearson_training, pearson_test):

        # experiment_params
        training_params = dict()
        if hasattr(self.estimator, 'cv_results_'):
            training_params['estimator'] = {
                'cv_results': pd.DataFrame(self.estimator.cv_results_).to_dict(),
                'best_estimator': self.estimator.best_estimator_.get_params(),
                'best_score': self.estimator.best_score_,
                'best_params': self.estimator.best_params_
            }
        else:
            training_params['estimator'] = self.estimator.get_params()

        training_params['estimator'].update({'estimator_name': self.estimator.__class__.__name__})

        with open(os.path.join(f'{experiment_path}_training_parameters.json'), 'w+') as fp:
            json.dump(training_params, fp)

        if hasattr(self.estimator, 'cv_results_'):
            print(training_params['estimator']['best_params'])
            best_results = dict()
            best_results.update(training_params['estimator']['best_estimator'])
            best_results.update({'best_score': [training_params['estimator']['best_score']]})
            best_results.update(training_params['estimator']['best_params'])
            best_results.update({'pearson_training': [pearson_training],
                                 'pearson_test': [pearson_test]})

            best_results_name = os.path.join(NLP_EXPERIMENT_PATH, f'{str(self.estimator.estimator.__class__.__name__)}_estimator_list.csv')
            df_best_results = pd.DataFrame(best_results)
            if not os.path.isfile(best_results_name):
                df_best_results.to_csv(best_results_name, header=list(df_best_results.columns), index=False)
            else:  # else it exists so append without writing the header
                df_best_results.to_csv(best_results_name, mode='a', header=False, index=False)

class Predicting:

    def __init__(self, estimator):
        self.estimator = estimator

    def __call__(self, X_test):
        return self.estimator.predict(X_test)


class Evaluating:

    def __init__(self, evaluators: Union[Evaluator, List[Evaluator]]):

        self.evaluators = evaluators

        if type(self.evaluators) is Evaluator:
            self.evaluators = [self.evaluators]

    def __call__(self, y_test, y_test_predicted, test_index, rsa_a_test, rsa_b_test):
        if self.evaluators is not None:
            for evaluator in self.evaluators:
                evaluator.evaluate(y_test, y_test_predicted, test_index, rsa_a_test, rsa_b_test)


if __name__ == '__main__':
    pass
