"""
.. module:: text_processing_methods
   :synopsis: Holding processing classes!
.. moduleauthor:: Klaus Kades
"""
import os
import re
from abc import ABC, abstractmethod
from typing import List

import gensim
import pandas as pd
import numpy as np
from nltk.util import ngrams
from pattern.en import parse
from difflib import SequenceMatcher
from pycontractions import Contractions
from segtok.segmenter import split_single
from segtok.tokenizer import split_contractions
from segtok.tokenizer import word_tokenizer

from mtc.helpers.text2digits import Text2Digits
from mtc.settings import NLP_MODELS_PATH
from mtc.mtc_properties import STOP_WORDS


def TextProcessing(name, *args, **kwargs):
    """
    All text processing classes should be called via this method
    """
    for cls in TextProcessingBaseClass.__subclasses__():
        if cls.__name__ == name:
            return cls(*args, **kwargs)
    raise ValueError('No text processing named %s' % name)


class TextProcessingBaseClass(ABC):
    """
    Any text processing class must inherit from this class
    """

    def preprocess(self,  sentence_list: List):

        if self.processing_type == 'corpus':
            return self._process_internal(sentence_list)
        else:
            processed_texts = []
            for text_element in sentence_list:
                text_element = self._process_internal(text_element)
                processed_texts.append(text_element)
            return processed_texts

    @property
    def processing_type(self):
        return 'sentence_level'

    @abstractmethod
    def level(self) -> str:
        pass

    @abstractmethod
    def _process_internal(self, sentence_list: List[str]) -> List[str]:
        """Private method to process a piece of text"""
        pass


class ContractionExpander(TextProcessingBaseClass):
    '''
    Removes contractions from the text and uses the full version instead (unification).

    Example:
    I'll walk down the road --> I will walk down the road
    '''

    model_contraction_expander = None

    def __init__(self, model=None):
        '''
        :param model: Pretrained word embedding model.
        '''
        super().__init__()

        if model is None:
            # If no model is given, use the default one and store it as a static class variable to avoid multiple loadings
            if ContractionExpander.model_contraction_expander is None:
                model_path = os.path.join(NLP_MODELS_PATH, 'pretrained', 'word_embeddings', 'pubmed2018_w2v_400D',
                                  'pubmed2018_w2v_400D.bin')
                ContractionExpander.model_contraction_expander = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

            model = ContractionExpander.model_contraction_expander

        self.cont = Contractions(kv_model=model)

    def level(self) -> str:
        return "text"

    def _process_internal(self, text: str) -> str:
        '''
        :param text: Input string.
        :return: The string without contractions.
        '''
        return list(self.cont.expand_texts([text], precise=True))[0]


class NumberUnifier(TextProcessingBaseClass):
    '''
    Unifies number representations in the text.

    Example:
    One capsule mouth two times day --> 1 capsule mouth 2 times day
    '''

    def __init__(self):
        super().__init__()
        self.t2n = Text2Digits()

    def level(self) -> str:
        return "text"

    def _process_internal(self, text: str) -> str:
        '''
        :param text: Input string.
        :return: The string without any numbers written-out.
        '''
        text = re.sub(r'(?:one)?-?half\b', '0.5', text)
        text = re.sub(r'(\d)\s*(?:to|-)\s*(\d)', r'\1-\2', text)
        text = self.t2n.convert(text)

        return text


class SpellingCorrector(TextProcessingBaseClass):
    '''
    Tries to correct some common spelling errors specific to the sts training dataset.

    Note that the case of the spell-corrected words is not preserved.

    Example:
        Pleast let me know --> Please let me know
    '''

    def __init__(self):
        self.errors = {
            'Pleast': 'please',
            'Locaation': 'location',
            'CURRENTand': 'current and',
            'LocationEmergency': 'location emergency',
            'refil': 'refill',
            'EDUCATIONReady': 'education ready',
            'd aily': 'daily'
        }

    def level(self) -> str:
        return "text"

    def _process_internal(self, text: str) -> str:
        for error, correction in self.errors.items():
            text = re.sub(r'\b' + error + r'\b', correction, text)

        return text


class RemoveActiveIngredients(TextProcessingBaseClass):
    '''
    Removes active ingredients: i.e: aw [aber] -> aw
    '''

    def level(self) -> str:
        return "text"

    def _process_internal(self, text: str) -> str:
        text = re.sub(r'\[([^]]+)\]', '', text)
            #text = re.sub(r'\b' + error + r'\b', correction, text)
        return text


class SentenceTokenizer(TextProcessingBaseClass):
    '''
    Tokenizes a piece of text into sentences. Note that this does also remove the whitespace between the sentences.

    Example:
        Hello there. How are you? --> [Hello there., How are you?]
    '''

    def level(self) -> str:
        return "text"

    def _process_internal(self, text: str) -> List[str]:
        return split_single(text)


class WordTokenizer(TextProcessingBaseClass):
    '''
    Tokenizes sentences into words.

    Example:
        [Hello there., How are you?] --> [[Hello, there, .], [How, are, you, ?]]
    '''

    def level(self) -> str:
        return "sentence"

    def _process_internal(self, sentences: List[str]) -> List[List[str]]:
        return [split_contractions(word_tokenizer(sen)) for sen in sentences]


class PunctuationRemover(TextProcessingBaseClass):
    '''
    Removes punctuation tokens.

    Example:
        [[Hello, there, .], [How, are, you, ?]] --> [[Hello, there], [How, are, you]]
    '''

    def level(self) -> str:
        return "token"

    def _is_punctuation(self, token):
        if re.search(r'\d-\d', token):
            # Keep number ranges
            return False
        elif re.search(r'\d\.', token) or re.search(r'\.\d', token):
            # Keep numbers
            return False
        else:
            return bool(re.search(r'[^a-zA-Z0-9äÄöÖüÜß\s]', token))

    def _process_internal(self, tokenized_sentences: List[List[str]]) -> List[List[str]]:
        sentences = []

        for sen in tokenized_sentences:
            words = [token for token in sen if not self._is_punctuation(token)]
            if words:
                sentences.append(words)

        return sentences


class MedicationRemover(TextProcessingBaseClass):
    '''
    '''

    def level(self) -> str:
        return "text"

    def _process_internal(self, text: str) -> str:
        return re.sub(r'\[[^]]+\]', '', text)


class DrugsStandardize(TextProcessingBaseClass):
    '''
    Removes punctuation tokens.

    Example:
        [['lasix']] --> ['FUROSEMIDE']
    '''

    def level(self) -> str:
        return "token"

    def _process_internal(self, tokenized_sentences: List[List[str]]) -> List[List[str]]:
        import mtc.drugstandards as drugs
        sentences = []

        for tokens in tokenized_sentences:
            std_tokens = drugs.standardize(tokens, thresh=1)

            std_sentence = []
            for std_token, token in zip(std_tokens, tokens):
                if std_token is None:
                    std_sentence.append(token)
                else:
                    std_sentence.append(std_token)

            sentences.append(std_sentence)

        return sentences


class StopWordsRemover(TextProcessingBaseClass):
    '''
    Removes stop word tokens.

    Example:
        [[Hello, there], [How, are, you]] --> [[Hello]]
    '''

    def level(self) -> str:
        return "token"

    def _process_internal(self, tokenized_sentences: List[List[str]]) -> List[List[str]]:
        sentences = []

        for sen in tokenized_sentences:
            words = [word for word in sen if word not in STOP_WORDS]
            if words:
                sentences.append(words)

        return sentences


class LowerCaseTransformer(TextProcessingBaseClass):
    '''
    Transforms every string to lowercase.

    Example:
        [[Hello]] --> [[hello]]
    '''

    def level(self) -> str:
        return "text or token"

    def _process_internal(self, text):
        if type(text) == str:
            return text.lower()
        else:
            sentences = []

            for sen in text:
                words = [word.lower() for word in sen]
                sentences.append(words)

            return sentences


class Lemmatizer(TextProcessingBaseClass):
    '''
    Lemmatizes tokens, i.e. transforms every word to its base form.

    Example:
        [[obtains]] --> [[obtain]]
    '''

    def level(self) -> str:
        return "token"

    def _process_internal(self, tokenized_sentences: List[List[str]]) -> List[List[str]]:
        sentences = []

        for sen in tokenized_sentences:
            # Unfortunately, the lemma() method does not yield the same result as parse (try e.g. with 'mice')
            words = [parse(token, lemmata=True).split('/')[-1] for token in sen]
            sentences.append(words)

        return sentences


class RemoveLowHighFrequencyWords(TextProcessingBaseClass):
    '''
    Removes words of high or low frequency.
    :param frequency: Number of occurrences.
    :param low_frequency: Low or high frequency
    '''

    def __init__(self, frequency: int, low_frequency: str):
        self.frequency = frequency
        self.low_frequency = low_frequency

    @property
    def processing_type(self):
        return 'corpus'

    def level(self) -> str:
        return "token"

    def _process_internal(self, sentence_list: List) -> List:
        print('up to know did not help a lot')
        new_sentence_list = []
        all_tokens = []

        for tokenized_sentences in sentence_list:
            for sen in tokenized_sentences:
                all_tokens = all_tokens + sen

        df_all_tokens = pd.DataFrame(all_tokens, columns=['tokens'])
        df_token_count = df_all_tokens['tokens'].value_counts()
        if self.low_frequency == 'low':
            print('attention, does not improve anything!')
            df_token_count = df_token_count[df_token_count <= self.frequency]
        elif self.low_frequency == 'high':
            df_token_count = df_token_count[df_token_count >= self.frequency]
        else:
            raise ValueError('low_frequency has to to be set to high or low')
        drop_tokens = df_token_count.index.to_list()

        print(drop_tokens)
        for tokenized_sentences in sentence_list:
            sentence = []
            for sen in tokenized_sentences:
                sentence.append([tok for tok in sen if tok not in drop_tokens])
            new_sentence_list.append(sentence)

        return new_sentence_list


class CondenseFrequentWordPairs(TextProcessingBaseClass):
    '''
    Connects words with an underscore that appear with a certain number of frequencies
    :param frequency: Number of occurrences.
    :param ngrams_num: Search for number of words that occur together
    '''

    @staticmethod
    def _subfinder(mylist, pattern):
        matches = []
        for i in range(len(mylist)):
            print(i)
            if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
                matches.append(pattern)
        return matches

    def __init__(self, frequency: int, ngrams_num: int):
        self.frequency = frequency
        self.ngrams_num = ngrams_num

    @property
    def processing_type(self):
        return 'corpus'

    def level(self) -> str:
        return "token"

    def _remove_duplictes(self, grams_words):
        no_drop_list = []
        replaced_list = []

        for search in grams_words:
            no_drop = True
            word_to_replace = search.split('_')
            for word in word_to_replace:
                if word in replaced_list:
                    no_drop = False
            if no_drop is True:
                replaced_list = replaced_list + word_to_replace
            no_drop_list.append(no_drop)
        return no_drop_list

    def _process_internal(self, sentence_list: List) -> List:
        print('up to know did not help a lot, due to wrong implementation, condenses too much!')
        new_sentence_list = []
        all_tokens = []

        for tokenized_sentences in sentence_list:
            for sen in tokenized_sentences:
                all_tokens = all_tokens + sen

        ngrams_list = list(ngrams(all_tokens, self.ngrams_num))
        df_ngrams = pd.DataFrame(['_'.join(n) for n in ngrams_list], columns=['ngrams'])
        df_counts = pd.DataFrame(df_ngrams['ngrams'].value_counts())
        df_filtered_counts = df_counts[df_counts['ngrams'] >= self.frequency]
        df_filtered_counts = df_filtered_counts[self._remove_duplictes(df_filtered_counts.index)]

        for tokenized_sentences in sentence_list:
            sentence = []
            for sen in tokenized_sentences:
                for search in df_filtered_counts.index.to_list():
                    words_to_replace = search.split('_')

                    sen = ' '.join(sen).replace(' '.join(words_to_replace), search).split()
                sentence.append(sen)
            new_sentence_list.append(sentence)

        return new_sentence_list


class RemoveFrequentWordPairs(TextProcessingBaseClass):
    '''
    Connects words with an underscore that appear with a certain number of frequencies
    :param frequency: Number of occurrences.
    :param ngrams_num: Search for number of words that occur together
    '''

    @staticmethod
    def _subfinder(mylist, pattern):
        matches = []
        for i in range(len(mylist)):
            print(i)
            if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
                matches.append(pattern)
        return matches

    def __init__(self, frequency: int, ngrams_num: int):
        self.frequency = frequency
        self.ngrams_num = ngrams_num

    @property
    def processing_type(self):
        return 'corpus'

    def level(self) -> str:
        return "token"

    def _remove_duplictes(self, grams_words):
        no_drop_list = []
        replaced_list = []

        for search in grams_words:
            no_drop = True
            word_to_replace = search.split('_')
            for word in word_to_replace:
                if word in replaced_list:
                    no_drop = False
            if no_drop is True:
                replaced_list = replaced_list + word_to_replace
            no_drop_list.append(no_drop)
        return no_drop_list

    def _process_internal(self, sentence_list: List) -> List:
        print('up to know did not help a lot, due to wrong implementation, condenses too much!')
        new_sentence_list = []
        all_tokens = []

        for tokenized_sentences in sentence_list:
            for sen in tokenized_sentences:
                all_tokens = all_tokens + sen

        ngrams_list = list(ngrams(all_tokens, self.ngrams_num))
        df_ngrams = pd.DataFrame(['_'.join(n) for n in ngrams_list], columns=['ngrams'])
        df_counts = pd.DataFrame(df_ngrams['ngrams'].value_counts())
        df_filtered_counts = df_counts[df_counts['ngrams'] >= self.frequency]
        df_filtered_counts = df_filtered_counts[self._remove_duplictes(df_filtered_counts.index)]

        for tokenized_sentences in sentence_list:
            sentence = []
            for sen in tokenized_sentences:
                for search in df_filtered_counts.index.to_list():
                    words_to_remove = search.split('_')

                    sen = ' '.join(sen).replace(' '.join(words_to_remove), '').split()
                sentence.append(sen)
            new_sentence_list.append(sentence)

        return new_sentence_list


class RemoveHighNGrams(TextProcessingBaseClass):
    '''
    Connects words with an underscore that appear with a certain number of frequencies
    :param frequency: Number of occurrences.
    :param ngrams_num: Search for number of words that occur together
    '''

    def __init__(self, ngrams_num: int, n_min: int):
        self.ngrams_num = ngrams_num
        self.n_min = n_min

    @property
    def processing_type(self):
        return 'corpus'

    def level(self) -> str:
        return "token"

    def _process_internal(self, sentence_list: List) -> List:
        print('up to know did not help a lot, due to wrong implementation, condenses too much!')

        # split sentences first to a and b
        sent_list_a = sentence_list[:len(sentence_list) // 2]
        sent_list_b = sentence_list[len(sentence_list) // 2:]

        new_sentence_list_a = []
        new_sentence_list_b = []
        total_shortend_sentences = []
        for idx, (sent_a, sent_b) in enumerate(zip(sent_list_a, sent_list_b)):
            sent_a = [item for sublist in sent_a for item in sublist]
            sent_b = [item for sublist in sent_b for item in sublist]
            s = SequenceMatcher(None, sent_a, sent_b)
            m = s.find_longest_match(0, len(sent_a), 0, len(sent_b))
            if m.size > self.ngrams_num:
                to_add_a = sent_a[:m.a] + sent_a[m.a+m.size:]
                to_add_b = sent_b[:m.b] + sent_b[m.b+m.size:]
                if len(to_add_a) <= self.n_min or len(to_add_b) <= self.n_min:
                    to_add_a = sent_a
                    to_add_b = sent_b
                else:
                    total_shortend_sentences.append(idx)

            else:
                to_add_a = sent_a
                to_add_b = sent_b
            new_sentence_list_a.append([to_add_a])
            new_sentence_list_b.append([to_add_b])
        print(total_shortend_sentences)
        print(len(total_shortend_sentences))
        return new_sentence_list_a + new_sentence_list_b

def stem_words():
    pass


def map_semantic_dictionary():
    pass
