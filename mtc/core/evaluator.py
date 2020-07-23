"""
.. module:: evaluator
   :synopsis: Holding all evaluator classes!
.. moduleauthor:: Klaus Kades
"""

from typing import List, Union, Dict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import classification_report

from mtc.core.sentence import Sentence


def Evaluator(name, *args, **kwargs):
    """
    All evaluator classes should be called via this method
    """
    for cls in EvaluatorBaseClass.__subclasses__():
        if cls.__name__ == name:
            return cls(*args, **kwargs)
    raise ValueError('No evalutor named %s' % name)


class EvaluatorBaseClass(ABC):
    """
    Any evaluator class must inherit from this class
    """

    @property
    def key_name(self):
        """Name must be unique!"""
        return self.__class__.__name__

    def evaluate(self,  *args, **kwargs) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        # if only one sentence is passed, convert to list of sentence

        self._evaluate_internal(*args, **kwargs)

    @abstractmethod
    def _evaluate_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        pass


class PearsonCorrelationCoefficientEvaluator(EvaluatorBaseClass):

    def __init__(self):
        super().__init__()
        self.results = dict()

    @property
    def key_name(self):
        """Name must be unique!"""
        return f"{self.__class__.__name__}"

    def _evaluate_internal(self, y_eval, y_eval_predicted, *args, **kwargs):

        # y_train = np.take(exp_data['y'], exp_data['idx_train'], axis=0)
        # y_pred_train = np.take(exp_data['y_pred'], exp_data['idx_train'], axis=0)
        # y_test = np.take(exp_data['y'], exp_data['idx_dev'], axis=0)
        # y_pred_test = np.take(exp_data['y_pred'], exp_data['idx_dev'], axis=0)
        self.results['pearson'] = [pearsonr(y_eval_predicted, y_eval)[0]]
        # self.results['pearson_test_set'] = [pearsonr(y_pred_test, y_test)[0]]
        # print('on training set with pcc: %f' % self.results['pearson'][0])
        print('PCC: %f' % self.results['pearson'][0])

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = self.key_name
        params['append'] = True
        params.update(self.results)
        return params


class PredictionAccuracyBySentence(EvaluatorBaseClass):

    def __init__(self):
        super().__init__()
        self.results = None
        self.diff_dict = {}

    @property
    def key_name(self):
        """Name must be unique!"""
        return f"{self.__class__.__name__}"

    def _evaluate_internal(self, y_eval, y_eval_predicted, test_index, rsa_a_eval, rsa_b_eval):

        self.diff_dict = {
            'diff': list(abs(y_eval-y_eval_predicted)),
            'sen_idx': test_index,
            'gold_standard': y_eval,
            'pred': y_eval_predicted,
            'raw_sentences_a': rsa_a_eval,
            'raw_sentences_b': rsa_b_eval
        }



    def get_params(self) -> Dict:
        params = dict()
        params['name'] = self.key_name
        params['append'] = False
        params['diff_dict'] = self.diff_dict

        return params


if __name__ == '__main__':
    from mtc.core.preprocessor import Preprocessor
    # from mtc.core.sentence import Sentence
    # from sklearn import linear_model, ensemble
    #
    # preprocessor = Preprocessor('DefaultPreprocessor')
    #
    # sentence_a = [
    #     Sentence('Hallo du, wie geht es dir?', preprocessor, {'ground_truth': 3}),
    #     Sentence('Mein Name ist Tina.', preprocessor, {'ground_truth':2})
    #     ]
    # sentence_b = [
    #     Sentence('Hi du, wie geht\'s?', preprocessor),
    #     Sentence('Mein Name ist Paul', preprocessor),
    #     ]
    #
    # clf = linear_model.Lasso(alpha=0.1)
    # classifier = Classifier('SelectiveClassifier', clf=clf, classifier_methods=[{'method': 'sequence_matcher_similarity'}])
    # evaluator = Evaluator('PCCE')
    #
    # classifier.fit(sentence_a, sentence_b)
    # classifier.predict(sentence_a, sentence_b)
    # evaluator.evaluate(sentence_a[0])
