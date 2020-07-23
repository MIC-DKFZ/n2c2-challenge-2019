"""
.. module:: training
   :synopsis: Holding all training classes!
.. moduleauthor:: Klaus Kades
"""

import os
from typing import List, Union, Dict
from abc import ABC, abstractmethod

from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from mtc.core.sentence import Sentence
from mtc.settings import NLP_MODELS_PATH
from mtc.helpers.decorators import convert_sentence_to_list


def EmbeddingTraining(embedding_name, *args, **kwargs):
    """
    All training classes should be called via this method
    """
    for cls in EmbeddingTrainingBaseClass.__subclasses__():
        if cls.is_training_for(embedding_name):
            return cls(*args, **kwargs)
    raise ValueError('No embedding trainer named %s' % embedding_name)


class EmbeddingTrainingBaseClass(ABC):
    """
    Any training class must inherit from this class
    """

    @classmethod
    @abstractmethod
    def is_training_for(cls, preprocessor_name):
        pass

    @abstractmethod
    def train(self, sentences: List[Sentence]):
        """Private method for adding embeddings to all words in a list of sentences."""
        pass

    def save_model(self):
        pass


class Word2VecTraining(EmbeddingTrainingBaseClass):

    @classmethod
    def is_training_for(cls, training_name):
        return training_name == 'Word2VecTraining'

    def __init__(self, new_model_name, pretrained_model_path=None, model_dict: Dict = None, train_dict: Dict = None, build_dict: Dict=None):
        super().__init__()

        if model_dict is None:
            model_dict = dict()

        if train_dict is None:
            train_dict = dict()

        if build_dict is None:
            build_dict = dict()

        self.new_model_name = new_model_name
        self.model_dir = os.path.join(NLP_MODELS_PATH, 'trained', 'word_embeddings', self.new_model_name)
        os.mkdir(self.model_dir)

        self.train_dict = train_dict
        self.build_dict = build_dict

        if pretrained_model_path is None:
            self.model = Word2Vec(**model_dict)
        else:
            self.model = Word2Vec.load(os.path.join(NLP_MODELS_PATH, pretrained_model_path))

    @convert_sentence_to_list
    def train(self, sentences: Union[List[Sentence], Sentence]):
        sentence_texts = []
        for sentence in sentences:
            sentence_texts.append(sentence.to_string_tokens())

        self.model.build_vocab(sentence_texts, **self.build_dict)
        self.model.train(sentence_texts, total_examples=self.model.corpus_count, **self.train_dict)

    def save_model(self):
        self.model.save(os.path.join(self.model_dir, '%s.model' % self.new_model_name))
        self.model.wv.save(os.path.join(self.model_dir, '%s.keyed_vector.model' % self.new_model_name))


class FastTextTraining(EmbeddingTrainingBaseClass):

    @classmethod
    def is_training_for(cls, training_name):
        return training_name == 'FastTextTraining'

    def __init__(self, new_model_name, pretrained_model_path=None, model_dict: Dict = None, train_dict: Dict = None, build_dict: Dict=None):
        super().__init__()

        if model_dict is None:
            model_dict = dict()

        if train_dict is None:
            train_dict = dict()

        if build_dict is None:
            build_dict = dict()

        self.new_model_name = new_model_name
        self.model_dir = os.path.join(NLP_MODELS_PATH, 'trained', 'word_embeddings', self.new_model_name)
        os.mkdir(self.model_dir)

        self.train_dict = train_dict
        self.build_dict = build_dict

        if pretrained_model_path is None:
            self.model = FastText(**model_dict)
        else:
            self.model = FastText.load(os.path.join(NLP_MODELS_PATH, pretrained_model_path))

    @convert_sentence_to_list
    def train(self, sentences: Union[List[Sentence], Sentence]):
        sentence_texts = []
        for sentence in sentences:
            sentence_texts.append(sentence.to_string_tokens())

        self.model.build_vocab(sentence_texts, **self.build_dict)
        self.model.train(sentence_texts, total_examples=len(sentences), **self.train_dict)

    def save_model(self):
        self.model.save(os.path.join(self.model_dir, '%s.model' % self.new_model_name))
        self.model.wv.save(os.path.join(self.model_dir, '%s.keyed_vector.model' % self.new_model_name))


class Doc2VecTraining(EmbeddingTrainingBaseClass):

    @classmethod
    def is_training_for(cls, training_name):
        return training_name == 'Doc2VecTraining'

    def __init__(self, new_model_name, pretrained_model_path=None, model_dict: Dict = None, train_dict: Dict = None, build_dict: Dict=None):
        super().__init__()

        if model_dict is None:
            model_dict = dict()

        if train_dict is None:
            train_dict = dict()

        if build_dict is None:
            build_dict = dict()

        self.new_model_name = new_model_name
        self.model_dir = os.path.join(NLP_MODELS_PATH, 'trained', 'document_embeddings', self.new_model_name)
        os.mkdir(self.model_dir)

        self.train_dict = train_dict
        self.build_dict = build_dict

        if pretrained_model_path is None:
            self.model = Doc2Vec(**model_dict)
        else:
            self.model = Doc2Vec.load(os.path.join(NLP_MODELS_PATH, pretrained_model_path))

    @convert_sentence_to_list
    def train(self, sentences: Union[List[Sentence], Sentence]):
        sentence_texts = []
        for sentence in sentences:
            sentence_texts.append(sentence.to_string_tokens())
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentence_texts)]

        self.model = Doc2Vec()
        self.model.build_vocab(documents, **self.build_dict)
        self.model.train(documents, total_examples=len(sentences), **self.train_dict)

    def save_model(self):
        self.model.save(os.path.join(self.model_dir, '%s.model' % self.new_model_name))


if __name__ == '__main__':

    from mtc.core.sentence import Sentence
    sen = [Sentence('Jo what up'), Sentence('Hallo du. Wie geht es dir? Hallo was ist hier los')]
    embedding_training = EmbeddingTraining('Doc2VecTraining', 'tes5t.model', train_dict={'epochs': 10}, build_dict={'min_count': 1})
    embedding_training.train(sen)
    embedding_training.save_model()
