"""
.. module:: preprocessor
   :synopsis: Holding all preprocessing classes!
.. moduleauthor:: Klaus Kades
"""
import os
from typing import List, Union, Dict
from abc import ABC, abstractmethod

import gensim

import mtc.helpers.text_processing as tpm
from mtc.settings import NLP_MODELS_PATH
from mtc.helpers.util import PipelineDictArgument
from mtc.helpers.text_processing import TextProcessing


def Preprocessor(name, *args, **kwargs):
    """
    All preprocessor classes should be called via this method
    """
    for cls in PreprocessorBaseClass.__subclasses__():
        if cls.__name__ == name:
            return cls(*args, **kwargs)
    raise ValueError('No preprocessorg named %s' % name)


class PreprocessorBaseClass(ABC):
    """
    Any preprocessor class must inherit from this class
    """

    def preprocess(self,  raw_texts: Union[str, List[str]]) -> List[str]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        # if only one sentence is passed, convert to list of sentence
        if type(raw_texts) is str:
            raw_texts = [raw_texts]

        return self._preprocess_sentences_internal(raw_texts)

    @abstractmethod
    def _preprocess_sentences_internal(self, raw_texts: List[str]) -> List[str]:
        """Private method for adding embeddings to all words in a list of sentences."""
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        pass


class SelectivePreprocessor(PreprocessorBaseClass):

    def __init__(self, processings: Union[List[tpm.TextProcessingBaseClass], List[TextProcessing]] = None):
        super().__init__()

        if processings:
            if type(processings[0]) is PipelineDictArgument:
                processings = [TextProcessing(*text_processing['args'], **text_processing['kwargs']) for text_processing in processings]
        else:
            processings = []

        self.processings = processings

        # The output after processing the text must be a single string and no tokens or sentences. To ensure this, the preprocessed text elements must be joined together via whitespace and this requires knowledge of the level
        self.output_level = 'text'
        if any([type(processing).__name__ == 'SentenceTokenizer' for processing in processings]):
            self.output_level = 'sentence'
        if any([type(processing).__name__ == 'WordTokenizer' for processing in processings]):
            self.output_level = 'token'

    def _preprocess_sentences_internal(self, raw_texts: List[str]) -> List[str]:

        for processing in self.processings:
            raw_texts = processing.preprocess(raw_texts)

        processed_texts = []
        for i, text_element in enumerate(raw_texts):
            # Join all words/sentences via whitespace since this is the expected input for the Sentence object
            if self.output_level == 'sentence':
                # Sentence level
                text_element = ' '.join(text_element)
            elif self.output_level == 'token':
                # Word level
                sentences = [' '.join(words) for words in text_element]
                text_element = ' '.join(sentences)

            processed_texts.append(text_element)

        return processed_texts

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = SelectivePreprocessor.__name__
        params['processings'] = [type(processing).__name__ for processing in self.processings]
        return params


class DefaultPreprocessor(PreprocessorBaseClass):

    def __init__(self):
        super().__init__()

        self.preprocessor = Preprocessor('SelectivePreprocessor', [
            tpm.TextProcessing('SentenceTokenizer'),
            tpm.TextProcessing('WordTokenizer#')
        ])

    def _preprocess_sentences_internal(self, raw_texts: List[str]) -> List[str]:
        return self.preprocessor.preprocess()

    def get_params(self) -> Dict:
        params = dict()
        params['name'] = DefaultPreprocessor.__name__
        return params


if __name__ == '__main__':
    from mtc.core.sentence import Sentence
    raw_text = 'Hello, how are you? My name is Peter. I do not like the weather today. But how are you? I have 3 children. I mean how are things? I have also'
    raw_texts = [
        'Hello, how are you?',
        'My name is Peter. I do not like the weather today.',
        'But how are you? I have 3 children. I mean how are things? I have also'
    ]

    preprocessor = Preprocessor('SelectivePreprocessor', [
        tpm.TextProcessing('SentenceTokenizer'),
        tpm.TextProcessing('WordTokenizer'),
        #tpm.TextProcessing('RemoveLowHighFrequencyWords', 2, 'high'),
        #tpm.TextProcessing('PunctuationRemover'),
        #tpm.TextProcessing('CondenseFrequentWordPairs', 2, 2),

        # tpm.TextProcessing('StopWordsRemover'),
        # tpm.TextProcessing('LowerCaseTransformer'),
    ])
    converted_texts = preprocessor.preprocess(raw_texts)
    print(converted_texts)
    # sentence_list = []
    # for converted_text in converted_texts:
    #     sentence_list.append(Sentence(raw_text, preprocessor))
    #
    # for sentence in sentence_list:
    #     sentence.to_plain_string()
    #     print(sentence)
