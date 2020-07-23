"""
.. module:: measure
   :synopsis: Holding all measure classes!
.. moduleauthor:: Klaus Kades
"""

from typing import List, Union, Dict
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
import re
import os

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import textdistance
#from similarity.longest_common_subsequence import LongestCommonSubsequence
#from similarity.metric_lcs import MetricLCS
#from similarity.qgram import QGram

from mtc.helpers.decorators import convert_sentence_to_list, timeit
from mtc.helpers.VectorSimilarityHelper import VectorSimilarityHelper
from mtc.core.sentence import Sentence
from mtc.core.embeddings import TokenEmbeddings, DocumentEmbeddings


# todo: inclulde more similarity measure from gensim


def SimilarityMeasures(name, *args, **kwargs):
    """
    All similarity measure classes should be called via this method
    """
    for cls in SimilarityMeasureBaseClass.__subclasses__():
        if cls.__name__ == name:
            return cls(*args, **kwargs)
    raise ValueError('No measure named %s' % cls.__name__)


class SimilarityMeasureBaseClass(ABC):
    """
    Any measure class must inherit from this class
    """

    def __init__(self, pa_preprocessor):
        self.pa_preprocessor = pa_preprocessor

    def measure(self, sentences_a: Union[List[Sentence], Sentence], sentences_b: Union[List[Sentence], Sentence],
                *args, **kwargs) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""
        everything_embedded = True

        if type(sentences_a) is Sentence:
            sentences_a = [sentences_a]

        if type(sentences_b) is Sentence:
            sentences_b = [sentences_b]

        for sentence_a in sentences_a:
            if self.key_name[:10] == 'BertResult' or self.key_name[:15] == 'MedicationGraph':
                everything_embedded = False
            if self.key_name not in sentence_a.sentence_properties.keys():
                everything_embedded = False

        if not everything_embedded:
            self._add_measure_internal(sentences_a, sentences_b, *args, **kwargs)

    @property
    def key_name(self):
        """Name must be unique!"""
        return f'{self.__class__.__name__}_{str(self.pa_preprocessor)}'

    @abstractmethod
    def _add_measure_internal(self, sentences_a: List[Sentence], sentences_b: List[Sentence], *args, **kwargs):
        """Private method for adding embeddings to all words in a list of sentences."""
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        pass


class WMDDistance(SimilarityMeasureBaseClass):
    """
    cf. `here <https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.wmdistance.html#gensim.models.Word2Vec.wmdistance>_`
    """

    def __init__(self, pa_preprocessor, pa_token_embedding):
        self.pa_token_embedding = pa_token_embedding
        super().__init__(pa_preprocessor)

    @property
    def key_name(self):
        return f"{self.__class__.__name__ }_{str(self.pa_token_embedding)}_{str(self.pa_preprocessor)}'"

    def _add_measure_internal(self, sentences_a: List[Sentence], sentences_b: List[Sentence], *args, **kwargs):
        print('measur wmd', self.pa_token_embedding['args'], self.pa_token_embedding['kwargs'])
        from time import time
        t1 = time()
        token_embedding = TokenEmbeddings(*self.pa_token_embedding['args'], **self.pa_token_embedding['kwargs'])
        print('as', time()-t1)
        for sentence_a, sentence_b in zip(sentences_a, sentences_b):
            distance = token_embedding.precomputed_word_embeddings.wmdistance(sentence_a.to_string_tokens(),
                                                                                    sentence_b.to_string_tokens())
            sentence_a.add_sentence_property({self.key_name: distance})

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = self.key_name
        return params


class CosineSimilarity(SimilarityMeasureBaseClass):
    """

    """
    def __init__(self, pa_preprocessor, pa_document_embedding):
        self.pa_document_embedding = pa_document_embedding
        super().__init__(pa_preprocessor)

    @property
    def key_name(self):
        return f'{self.__class__.__name__ }_{str(self.pa_document_embedding)}_{str(self.pa_preprocessor)}'

    def _add_measure_internal(self, sentences_a: List[Sentence], sentences_b: List[Sentence], *args, **kwargs):

        document_embedding = DocumentEmbeddings(*self.pa_document_embedding['args'], **self.pa_document_embedding['kwargs'])

        document_embedding.embed_str(sentences_a)
        document_embedding.embed_str(sentences_b)

        for sentence_a, sentence_b in zip(sentences_a, sentences_b):
            similarity = cosine(sentence_a.get_embedding_by_name(document_embedding.name).tolist(),
                                sentence_b.get_embedding_by_name(document_embedding.name).tolist())
            sentence_a.add_sentence_property({self.key_name: similarity})

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = self.key_name
        return params


class VectorSimilarities(SimilarityMeasureBaseClass):
    """

    """
    def __init__(self, pa_preprocessor, pa_document_embedding, measure_name: str):
        self.pa_document_embedding = pa_document_embedding
        self.measure_name = measure_name
        self.sim_measure = VectorSimilarityHelper()
        super().__init__(pa_preprocessor)

    @property
    def key_name(self):
        return f'{self.__class__.__name__ }_{self.measure_name}_{str(self.pa_document_embedding)}_{str(self.pa_preprocessor)}'

    def _add_measure_internal(self, sentences_a: List[Sentence], sentences_b: List[Sentence], *args, **kwargs):

        document_embedding = DocumentEmbeddings(*self.pa_document_embedding['args'], **self.pa_document_embedding['kwargs'])

        document_embedding.embed_str(sentences_a)
        document_embedding.embed_str(sentences_b)

        for sentence_a, sentence_b in zip(sentences_a, sentences_b):
            if self.measure_name == 'euclidean':
                similarity = self.sim_measure.euclidean_distance(sentence_a.get_embedding_by_name(document_embedding.name).tolist(),
                                sentence_b.get_embedding_by_name(document_embedding.name).tolist())
            elif self.measure_name == 'manhattan':
                similarity = self.sim_measure.manhattan_distance(sentence_a.get_embedding_by_name(document_embedding.name).tolist(),
                                sentence_b.get_embedding_by_name(document_embedding.name).tolist())
            elif self.measure_name == 'minkowski':
                similarity = self.sim_measure.minkowski_distance(sentence_a.get_embedding_by_name(document_embedding.name).tolist(),
                            sentence_b.get_embedding_by_name(document_embedding.name).tolist(), 3)
            elif self.measure_name == 'cosine_similarity':
                similarity = self.sim_measure.cosine_similarity(sentence_a.get_embedding_by_name(document_embedding.name).tolist(),
                                       sentence_b.get_embedding_by_name(document_embedding.name).tolist())
            elif self.measure_name == 'jaccard_similarity':
                similarity = self.sim_measure.jaccard_similarity(sentence_a.get_embedding_by_name(document_embedding.name).tolist(),
                                       sentence_b.get_embedding_by_name(document_embedding.name).tolist())

            else:
                raise ValueError('similarity does not exists')
            sentence_a.add_sentence_property({self.key_name: similarity})

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = self.key_name
        return params


class SequenceMatcherSimilarity(SimilarityMeasureBaseClass):
    """
    """

    def __init__(self, pa_preprocessor):
        super().__init__(pa_preprocessor)

    def _add_measure_internal(self, sentences_a: List[Sentence], sentences_b: List[Sentence], *args, **kwargs):
        for sentence_a, sentence_b in zip(sentences_a, sentences_b):
            similar_word_num = SequenceMatcher(None, sentence_a.to_string_tokens(), sentence_b.to_string_tokens()).ratio()
            sentence_a.add_sentence_property({self.key_name: similar_word_num})

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = self.key_name
        return params


class IncludesMedicationSimilarity(SimilarityMeasureBaseClass):
    """
    Simply states if two sentences have a medicament or not
    """

    def __init__(self, pa_preprocessor):
        super().__init__(pa_preprocessor)

    def _add_measure_internal(self, sentences_a: List[Sentence], sentences_b: List[Sentence], *args, **kwargs):
        for sentence_a, sentence_b in zip(sentences_a, sentences_b):
            match_a = re.search(r'(\d+\.?\d*|\d*\.?\d+)\s*(mcg|mg|grams?|g|ml|liters?)\b', sentence_a.to_original_text())
            match_b = re.search(r'(\d+\.?\d*|\d*\.?\d+)\s*(mcg|mg|grams?|g|ml|liters?)\b', sentence_b.to_original_text())
            if (match_a is None and match_b is not None) or (match_a is not None and match_b is None):
                sentence_a.add_sentence_property({self.key_name: 0})
            else:
                sentence_a.add_sentence_property({self.key_name: 20})

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = self.key_name
        return params


class Textdistance(SimilarityMeasureBaseClass):
    """
    """

    def __init__(self, pa_preprocessor, name, qval=1):
        super().__init__(pa_preprocessor)

        self.time_log = []
        self.qval = qval
        self.textdistance_name = name

        # Edited based:
        if name == 'Hamming':
            self.similar_measure = textdistance.Hamming(qval=qval)
        elif name == 'DamerauLevenshtein':
            self.similar_measure = textdistance.DamerauLevenshtein(qval=qval)
        elif name == 'Levenshtein':
            self.similar_measure = textdistance.Levenshtein(qval=qval)
        elif name == 'Mlipns':
            self.similar_measure = textdistance.MLIPNS(qval=qval)
        elif name == 'Jaro':
            self.similar_measure = textdistance.Jaro(qval=qval)
        elif name == 'JaroWinkler':
            self.similar_measure = textdistance.JaroWinkler(qval=qval)
        elif name == 'StrCmp95':
            self.similar_measure = textdistance.StrCmp95()
        elif name == 'NeedlemanWunsch':
            self.similar_measure = textdistance.NeedlemanWunsch(qval=qval)
        elif name == 'Gotoh':
            self.similar_measure = textdistance.Gotoh(qval=qval)
        elif name == 'SmithWaterman':
            self.similar_measure = textdistance.SmithWaterman(qval=qval)

        # Token based
        elif name == 'Jaccard':
            self.similar_measure = textdistance.Jaccard(qval=qval)
        elif name == 'Sorensen':
            self.similar_measure = textdistance.Sorensen(qval=qval)
        elif name == 'Tversky':
            self.similar_measure = textdistance.Tversky()
        elif name == 'Overlap':
            self.similar_measure = textdistance.Overlap(qval=qval)
        elif name == 'Tanimoto':
            self.similar_measure = textdistance.Tanimoto(qval=qval)
        elif name == 'Cosine':
            self.similar_measure = textdistance.Cosine(qval=qval)
        elif name == 'MongeElkan':
            self.similar_measure = textdistance.MongeElkan(qval=qval)
        elif name == 'Bag':
            self.similar_measure = textdistance.Bag(qval=qval)

        # Sequence based
        elif name == 'LCSSeq':
            self.similar_measure = textdistance.LCSSeq(qval=qval)
        elif name == 'LCSStr':
            self.similar_measure = textdistance.LCSStr(qval=qval)
        elif name == 'RatcliffObershelp':
            self.similar_measure = textdistance.RatcliffObershelp(qval=qval)

        # Compression based
        elif name == 'ArithNCD':
            self.similar_measure = textdistance.ArithNCD(qval=qval)
        elif name == 'RLENCD':
            self.similar_measure = textdistance.RLENCD(qval=qval)
        elif name == 'BWTRLENCD':
            self.similar_measure = textdistance.BWTRLENCD()
        elif name == 'SqrtNCD':
            self.similar_measure = textdistance.SqrtNCD(qval=qval)
        elif name == 'EntropyNCD':
            self.similar_measure = textdistance.EntropyNCD(qval=qval)

        # Simple:
        elif name == 'Prefix':
            self.similar_measure = textdistance.Prefix(qval=qval)
        elif name == 'Postfix':
            self.similar_measure = textdistance.Postfix(qval=qval)
        elif name == 'Length':
            self.similar_measure = textdistance.Length(qval=qval)
        elif name == 'Identity':
            self.similar_measure = textdistance.Identity(qval=qval)
        elif name == 'Matrix':
            self.similar_measure = textdistance.Matrix()

    @property
    def key_name(self):
        return f'{self.__class__.__name__ }_{self.textdistance_name}_qval_{self.qval}_{str(self.pa_preprocessor)}'

    @timeit
    def _add_measure_internal(self, sentences_a: List[Sentence], sentences_b: List[Sentence], *args, **kwargs):
        print(self.key_name)
        for sentence_a, sentence_b in zip(sentences_a, sentences_b):
            try:
                distance = self.similar_measure.similarity(sentence_a.to_tokenized_string(), sentence_b.to_tokenized_string())
            except ZeroDivisionError:
                distance = 0
            sentence_a.add_sentence_property({self.key_name: distance})

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = self.key_name
        params['timings'] = self.time_log
        params['qval'] = self.qval
        return params


class Strtsim(SimilarityMeasureBaseClass):
    """ They only lead to bad results...
        Horrible performans!
    """

    def __init__(self, pa_preprocessor, name):
        super().__init__(pa_preprocessor)

        self.time_log = []
        self.textdistance_name = name

        # Edited based:
        if name == 'QGram':
            self.similar_measure = QGram()
        elif name == 'LongestCommonSubsequence':
            self.similar_measure = LongestCommonSubsequence()
        elif name == 'MetricLCS':
            self.similar_measure = MetricLCS()

    @property
    def key_name(self):
        return f'{self.__class__.__name__}_{self.textdistance_name}_{str(self.pa_preprocessor)}'

    @timeit
    def _add_measure_internal(self, sentences_a: List[Sentence], sentences_b: List[Sentence], *args, **kwargs):
        print(self.key_name)
        for sentence_a, sentence_b in zip(sentences_a, sentences_b):
            distance = self.similar_measure.distance(sentence_a.to_tokenized_string(),
                                                       sentence_b.to_tokenized_string())
            sentence_a.add_sentence_property({self.key_name: distance})

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = self.key_name
        params['timings'] = self.time_log
        params['qval'] = self.qval
        return params


class BertResult(SimilarityMeasureBaseClass):
    """
    """

    def __init__(self, pa_preprocessor):
        super().__init__(pa_preprocessor)

    def _add_measure_internal(self, sentences_a: List[Sentence], sentences_b: List[Sentence], train_or_test):

        print('Updating Bert results')

        if train_or_test == 'train':
            df_scores = pd.read_csv(os.path.join(os.environ.get('BERT_SCORES_PATH'), 'train_scores.csv'), header=None)
        elif train_or_test == 'test':
            df_scores = pd.read_csv(os.path.join(os.environ.get('BERT_SCORES_PATH'), 'test_scores.csv'), header=None)
        else:
            raise FileNotFoundError('train_or_test must either be train or test')

        for sentence_a, sentence_b, score in zip(sentences_a, sentences_b, df_scores[0]):
            sentence_a.add_sentence_property({self.key_name: score})

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = self.key_name
        return params


class MedicationGraph(SimilarityMeasureBaseClass):
    """
    """

    def __init__(self, pa_preprocessor):
        super().__init__(pa_preprocessor)

    def _add_measure_internal(self, sentences_a: List[Sentence], sentences_b: List[Sentence], train_or_test):
        print('updating medication graph')
        if train_or_test == 'train':
            df_scores = pd.read_csv(os.path.join('..', 'similarity_measure_lists', 'tablet_similarity_train.csv'))
        elif train_or_test == 'test':
            df_scores = pd.read_csv(os.path.join('..', 'similarity_measure_lists', 'tablet_similarity_test.csv'))
        else:
            raise FileNotFoundError('train_or_test must either be train or test')

        for sentence_a, sentence_b, score in zip(sentences_a, sentences_b, df_scores['score']):
            sentence_a.add_sentence_property({self.key_name: score})

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = self.key_name
        return params


class AddDocumentEmbedding(SimilarityMeasureBaseClass):
    """

    """
    def __init__(self, pa_preprocessor, pa_document_embedding):
        self.pa_document_embedding = pa_document_embedding
        super().__init__(pa_preprocessor)

    @property
    def key_name(self):
        return f'{self.__class__.__name__ }_{str(self.pa_document_embedding)}_{str(self.pa_preprocessor)}'

    def _add_measure_internal(self, sentences_a: List[Sentence], sentences_b: List[Sentence], *args, **kwargs):

        document_embedding = DocumentEmbeddings(*self.pa_document_embedding['args'], **self.pa_document_embedding['kwargs'])

        document_embedding.embed_str(sentences_a)
        document_embedding.embed_str(sentences_b)

        for sentence_a, sentence_b in zip(sentences_a, sentences_b):
            # all_document_embeddings = sentence_a.get_embedding_by_name(document_embedding.name).tolist() + \
            #                          sentence_b.get_embedding_by_name(document_embedding.name).tolist()
            all_document_embeddings = [x1 - x2 for (x1, x2) in
                                       zip(sentence_a.get_embedding_by_name(document_embedding.name).tolist(), sentence_b.get_embedding_by_name(document_embedding.name).tolist())]
            sentence_a.add_sentence_property({self.key_name: all_document_embeddings})

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = self.key_name
        return params


class TabletFeatures(SimilarityMeasureBaseClass):
    def __init__(self, pa_preprocessor):
        super().__init__(pa_preprocessor)

    @property
    def key_name(self):
        return f'{self.__class__.__name__ }_{str(self.pa_preprocessor)}'

    def _extract_tuple(self, sentence):
        def replace_range(match):
            num1 = match.group(1)
            num2 = match.group(2)

            return '{:.2f}'.format((float(num1) + float(num2)) / 2)

        # Replace ranges
        sentence = re.sub(r'(\d+\.?\d*|\d*\.?\d+)\s*-\s*(\d+\.?\d*|\d*\.?\d+)', replace_range, sentence)

        amount = 0  # In mg
        before_amount = ''
        match = re.search(r'(\d+\.?\d*|\d*\.?\d+)\s*(mcg|mg|grams?|g|ml|liters?)\b', sentence)
        if match:
            number = float(match.group(1))
            unit = match.group(2)

            before_amount = sentence[:match.start(0)]

            if unit == 'mg':
                amount = number
            elif unit == 'mcg':
                amount = number / 1000
            else:
                # gram
                amount = number * 1000

        frequency = 0
        match = re.search(r'(\d+\.?\d*|\d*\.?\d+)\s+(?:tablets?|capsules?|packages?|puffs?)', sentence)
        if match:
            frequency = float(match.group(1))

        dose = 0  # In number of times per day
        match1 = re.search(r'(\d+\.?\d*|\d*\.?\d+)\s+(times?\s*(?:daily)?|hours?)', sentence)
        match2 = re.search(r'every\s+(?:bedtime|evening)', sentence)
        if match1:
            number = float(match1.group(1))
            unit = match1.group(2)

            if 'hour' in unit and number != 0:
                # E.g. every 4 hours --> 3 times per day (assuming that tablet are not supposed to be taken at night)
                dose = 12 / number
            else:
                dose = number
        elif match2:
            dose = 1

        return amount, frequency, dose

    def _add_measure_internal(self, sentences_a: List[Sentence], sentences_b: List[Sentence], *args, **kwargs):
        for sentence_a, sentence_b in zip(sentences_a, sentences_b):
            features_a = self._extract_tuple(sentence_a.to_plain_string())
            features_b = self._extract_tuple(sentence_b.to_plain_string())
            diff = np.array(features_a) - np.array(features_b)
            sentence_a.add_sentence_property({self.key_name: list(abs(np.array(list(features_a)) - np.array(list(features_b))))})

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = self.key_name
        return params


if __name__ == '__main__':
    from mtc.core.sentence import Sentence
    from mtc.core.preprocessor import Preprocessor
    from mtc.helpers.util import PipelineDictArgument

    pa_preprocessor1 = PipelineDictArgument('SelectivePreprocessor', [
        PipelineDictArgument('NumberUnifier'),
        PipelineDictArgument('SpellingCorrector'),
        PipelineDictArgument('SentenceTokenizer'),
        PipelineDictArgument('WordTokenizer'),
        PipelineDictArgument('PunctuationRemover'),
        PipelineDictArgument('StopWordsRemover'),
        PipelineDictArgument('LowerCaseTransformer'),
        PipelineDictArgument('Lemmatizer'),
    ])
    textdistance_names = [
        'Hamming',
        'DamerauLevenshtein',
        'Levenshtein',
        # 'Mlipns', # todo: does not work
        'Jaro',
        'JaroWinkler',
        'NeedlemanWunsch',
        # 'Gotoh',# takes forever
        'SmithWaterman',
        # 'Jaccard',
        # 'Sorensen',
        # 'Tversky',
        # 'Overlap',
        # 'Tanimoto',
        # 'Cosine',
        # 'MongeElkan',
        # 'Bag',
        # 'LCSSeq',
        # 'LCSStr',
        # 'RatcliffObershelp',
    ]

    #
    measures_textdistance1 = [SimilarityMeasures('Textdistance', pa_preprocessor1, measure, qval=1) for measure in
                              textdistance_names]
    measures_textdistance2 = [SimilarityMeasures('Textdistance', pa_preprocessor1, measure, qval=2) for measure in
                              textdistance_names]
    measures_textdistance3 = [SimilarityMeasures('Textdistance', pa_preprocessor1, measure, qval=3) for measure in
                              textdistance_names]

    sentences_a = [Sentence('Hi du')]
    sentences_b = [Sentence('My name is Emmy')]
    measure = SimilarityMeasures('Textdistance', 'Levenshtein', qval=2)


