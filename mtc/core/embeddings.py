"""
.. module:: embeddings
   :synopsis: Holding all embedding classes!
.. moduleauthor:: Klaus Kades
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import logging
import copy

import torch
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import flair
from flair.embeddings import \
    Embeddings as FlairBaseEmbeddings, \
    DocumentEmbeddings as FlairDocumentEmbeddings, \
    TokenEmbeddings as FlairTokenEmbeddings, \
    WordEmbeddings as FlairWordEmbeddings, \
    StackedEmbeddings as FlairStackedEmbeddings,\
    CharacterEmbeddings as FlairCharacterEmbeddings,\
    FlairEmbeddings as FlairFlairEbmeddings,\
    BytePairEmbeddings as FlairBytePairEmbeddings,\
    DocumentPoolEmbeddings as FlairDocumentPoolEmbeddings,\
    DocumentRNNEmbeddings as FlairDocumentRNNEmbeddings,\
    DocumentLMEmbeddings as FlairDocumentLMEmbeddings
from flair.data import Token
from gensim.models import Doc2Vec
from bert_serving.client import BertClient

from mtc.core.sentence import Sentence
from mtc.helpers.util import chunks, get_token_embedding_by_names, PipelineDictArgument
from mtc.helpers.decorators import convert_sentence_to_list
from mtc.settings import PATH_TO_INFERSENT, NLP_MODELS_PATH
from mtc.mtc_properties import LOADED_EMBEDDINGS


# import InferSent
sys.path.insert(0, PATH_TO_INFERSENT)
from models import InferSent

log = logging.getLogger('flair')


class EmbeddingBaseClass(ABC):
    """
    **Base class for embeddings:**
    All embedding classes must additionally inherit from flair.embeddings.TokenEmbeddings or flair.embeddings.DocumentEmbeddings
    All embeddings should define a self.name!
    """

    @convert_sentence_to_list
    def embed_str(self, sentences: Union[Sentence, List[Sentence]]):
        """
        Necessary, since embed function in flair ask for type(sentence), however type(sentence) is mtc.core.sentence.
        Sentence not flair.data.Sentence anymore! to make list and sentence as arguments possible embed_str should
        always come with the convert_sentence_to_list decorator!
        """
        FlairBaseEmbeddings.embed(self, sentences)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'' Word embeddings
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def TokenEmbeddings(name, *args, **kwargs):
    """
    All token embedding classes should be called via this method
    """
    for cls in EmbeddingBaseClass.__subclasses__():
        if cls.__name__ == name:
            return LOADED_EMBEDDINGS.get_class_instance(cls, name, *args, **kwargs)
    raise ValueError('No token embedding named %s' % name)


class StackedEmbeddings(EmbeddingBaseClass, FlairStackedEmbeddings):
    """
    This class extends the StackEmbeddings class from Flair, cf.
    `here <https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md>`_
    """

    def __init__(self, arg, *args, **kwargs):
        super(StackedEmbeddings, self).__init__(arg, *args, **kwargs)

    @convert_sentence_to_list
    def embed_str(self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool = True):
        self.embed(sentences, static_embeddings)


class WordEmbeddings(EmbeddingBaseClass, FlairWordEmbeddings):
    """
    This class extends the WordEmbeddings class from Flair. Word embeddings listed
    `here <https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md>`_
    are supported as well as any gensim embedding. They can be loaded by giving the total path of the gensim model to the constructor
    :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code or total path to custom gensim model
    """

    def __init__(self, *args, **kwargs):

        super(WordEmbeddings, self).__init__(*args, **kwargs)


class CharacterEmbeddings(EmbeddingBaseClass, FlairCharacterEmbeddings):
    """
    This class extends the CharacterEmbeddings class from Flair, cf.
    `here <https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md>`_
    """

    def __init__(self, arg, *args, **kwargs):
        super(CharacterEmbeddings, self).__init__(arg, *args, **kwargs)


class BytePairEmbeddings(EmbeddingBaseClass, FlairBytePairEmbeddings):
    """
    This class extends the BytePairEmbeddings class from Flair, cf.
    `here <https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md>`_
    """

    def __init__(self, arg, *args, **kwargs):
        super(BytePairEmbeddings, self).__init__(arg, *args, **kwargs)


class FlairEmbeddings(EmbeddingBaseClass, FlairFlairEbmeddings):
    """
    This class extends the FlairEmbeddings class from Flair, cf.
    `here <https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md>`_
    """

    def __init__(self, arg, *args, **kwargs):
        super(FlairEmbeddings, self).__init__(arg, *args, **kwargs)


class BertEmbeddings(EmbeddingBaseClass, FlairTokenEmbeddings):
    """
    Class to use bert_serving with flair sentence.

    To use it, start the server (helpers/bert_embedding_server_worker.py) with the desired model and start with the following example:

    from mtc.core.sentence import Sentence
    from mtc.core.embeddings import TokenEmbeddings

    sentence = Sentence('I check my bank account.')

    embedding = TokenEmbeddings('BertEmbeddings')
    embedding.embed_str(sentence)

    # The embeddings are then stored in the sentence object
    sentence.tokens[0].embedding
    """

    def __init__(self, bert_model_or_path: str = 'bert_12_768_12'):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """

        self.name: str = str(bert_model_or_path)
        self.static_embeddings = True

        self.bc = BertClient()
        # raise ValueError(f'The given embeddings "{embeddings}" is not available or is not a valid path.')
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return 768

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):
            tokenized_text = sentence.to_tokenized_string().split(' ')
            chunk_size = 126
            for tok_text, sen_tokens in zip(chunks(tokenized_text, chunk_size), chunks(sentence.tokens, chunk_size)):
                vec = self.bc.encode([tok_text], show_tokens=True, is_tokenized=True)

                # Remove first and last argument:
                be_token_vec = vec[0][0][1:-1]
                be_word_vec = vec[1][0][1:-1]

                for token, be_word, be_token in zip(sen_tokens, be_word_vec, be_token_vec):
                    token: Token = token
                    if be_word == token.text:
                        word_embedding = be_token
                    elif be_word == '[UNK]':
                        word_embedding = np.zeros(self.embedding_length, dtype='float')
                    else:
                        raise ValueError(f'There went something wrong during the BERT embedding.')

                    word_embedding = torch.FloatTensor(word_embedding)
                    token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'' Document embeddings
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def DocumentEmbeddings(name, *args, **kwargs):
    """
    All document embedding classes should be called via this method
    """
    for cls in EmbeddingBaseClass.__subclasses__():
        if cls.__name__ == name:
            return LOADED_EMBEDDINGS.get_class_instance(cls, name, *args, **kwargs)
    raise ValueError('No document embedding named %s' % name)


class DocumentPoolEmbeddings(EmbeddingBaseClass, FlairDocumentPoolEmbeddings):
    """
    This class extends the DocumentPoolEmbeddings class from Flair, cf.
    `here <https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md>`_
    Pooling can be done in three ways ['mean', 'max', 'min']
    """

    def __init__(self, embeddings: List, *args, **kwargs):
        if type(embeddings[0]) is PipelineDictArgument:
            embeddings = [TokenEmbeddings(*pa_token_embedding['args'], **pa_token_embedding['kwargs'])
                          for pa_token_embedding in embeddings]

        super(DocumentPoolEmbeddings, self).__init__(embeddings, *args, **kwargs)

        self.relevant_token_embeddings = [rel_em.name for rel_em in self.embeddings.embeddings]

        string_relevant_token_embeddings = "".join(self.relevant_token_embeddings)
        self.name = f"{self.__class__.__name__ }_{self.pooling}_{string_relevant_token_embeddings}"
        self.static_embeddings = True

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        """
        !! Copied from flair.embeddings.DocumentPoolEmbeddings.embed()
        :param sentences:
        :return:            return loaded_embeddings.get_class_instance(cls, name, *args, **kwargs)
        """
        self.embeddings.embed(sentences)

        for sentence in sentences:
            word_embeddings = []
            for token in sentence.tokens:
                ################################################################################
                # added, the rest is copied
                word_embeddings.append(get_token_embedding_by_names(token, self.relevant_token_embeddings).unsqueeze(0))
                ################################################################################

            word_embeddings = torch.cat(word_embeddings, dim=0).to(flair.device)

            if self.fine_tune_mode in ["nonlinear", "linear"]:
                word_embeddings = self.embedding_flex(word_embeddings)

            if self.fine_tune_mode in ["nonlinear"]:
                word_embeddings = self.embedding_flex_nonlinear(word_embeddings)
                word_embeddings = self.embedding_flex_nonlinear_map(word_embeddings)

            if self.pooling == "mean":
                pooled_embedding = self.pool_op(word_embeddings, 0)
            else:
                pooled_embedding, _ = self.pool_op(word_embeddings, 0)

            sentence.set_embedding(self.name, pooled_embedding)


class DocumentRNNEmbeddings(EmbeddingBaseClass, FlairDocumentRNNEmbeddings):
    """
    This class extends the DocumentRNNEmbeddings class from Flair, cf.
    `here <https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md>`_
    The following RNNs are supported: 'GRU', 'LSTM', 'RNN_TANH' or 'RNN_RELU'
    """

    def __init__(self, arg, *args, **kwargs):
        super(DocumentRNNEmbeddings, self).__init__(arg, *args, **kwargs)

        self.relevant_token_embeddings = [rel_em.name for rel_em in self.embeddings.embeddings]
        string_relevant_token_embeddings = "".join(self.relevant_token_embeddings)
        self.name = f"{self.__class__.__name__ }_{self.rnn._get_name()}_{string_relevant_token_embeddings}"

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        """
        !! Copied from flair.embeddings.DocumentRNNEmbeddings.embed()
        :param sentences:
        :return:

        """

        self.rnn.zero_grad()

        sentences.sort(key=lambda x: len(x), reverse=True)

        self.embeddings.embed(sentences)

        # first, sort sentences by number of tokens
        longest_token_sequence_in_batch: int = len(sentences[0])

        all_sentence_tensors = []
        lengths: List[int] = []

        # go through each sentence in batch
        for i, sentence in enumerate(sentences):

            lengths.append(len(sentence.tokens))

            word_embeddings = []

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                ################################################################################
                # added, the rest is copied
                word_embeddings.append(get_token_embedding_by_names(token, self.relevant_token_embeddings).unsqueeze(0))
                ################################################################################

            # PADDING: pad shorter sentences out
            for add in range(longest_token_sequence_in_batch - len(sentence.tokens)):
                word_embeddings.append(
                    torch.zeros(
                        self.length_of_all_token_embeddings, dtype=torch.float
                    ).unsqueeze(0)
                )

            word_embeddings_tensor = torch.cat(word_embeddings, 0).to(flair.device)

            sentence_states = word_embeddings_tensor

            # ADD TO SENTENCE LIST: add the representation
            all_sentence_tensors.append(sentence_states.unsqueeze(1))

        # --------------------------------------------------------------------
        # GET REPRESENTATION FOR ENTIRE BATCH
        # --------------------------------------------------------------------
        sentence_tensor = torch.cat(all_sentence_tensors, 1)

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        # use word dropout if set
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)

        sentence_tensor = self.dropout(sentence_tensor)

        packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor, lengths)

        self.rnn.flatten_parameters()

        rnn_out, hidden = self.rnn(packed)

        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_out)

        outputs = self.dropout(outputs)

        # --------------------------------------------------------------------
        # EXTRACT EMBEDDINGS FROM RNN
        # --------------------------------------------------------------------
        for sentence_no, length in enumerate(lengths):
            last_rep = outputs[length - 1, sentence_no]

            embedding = last_rep
            if self.bidirectional:
                first_rep = outputs[0, sentence_no]
                embedding = torch.cat([first_rep, last_rep], 0)

            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)


class DocumentLMEmbeddings(EmbeddingBaseClass, FlairDocumentLMEmbeddings):
    """
    This class extends the DocumentLMEmbeddings class from Flair, cf.
    `here <https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md>`_
    """

    def __init__(self, arg, *args, **kwargs):
        super(DocumentLMEmbeddings, self).__init__(arg, *args, **kwargs)


class Doc2VecEmbeddings(EmbeddingBaseClass, FlairDocumentEmbeddings):
    """
    Class to infer the gensmim Doc2Vec to flair sentences.
    """

    def __init__(self, doc_embedding: str, random_seed: int=None ):

        super().__init__()

        self.pretrained_model = Doc2Vec.load(doc_embedding)

        self._random_seed = random_seed
        if self._random_seed is not None:
            self.pretrained_model.random.seed(self._random_seed)

        self.__embedding_length: int = self.pretrained_model.vector_size
        self.name = f"{self.__class__.__name__}_{doc_embedding}"

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        everything_embedded = True
        for sentence in sentences:
            if self.name not in sentence._embeddings.keys(): everything_embedded = False

        if not everything_embedded:
            for sentence in sentences:
                sentence_embedding = torch.tensor(self.pretrained_model.infer_vector([t.text for t in sentence.tokens]))
                sentence.set_embedding(self.name, sentence_embedding)


class BertSentenceEmbeddings(EmbeddingBaseClass, FlairDocumentEmbeddings):
    """
    Class to infer the Google Universal Sentence Encoder to flair sentences.
    """

    def __init__(self):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """
        super().__init__()

        self.static_embeddings = True

        self.name = f"{self.__class__.__name__}"

        self.bc = BertClient()
        # raise ValueError(f'The given embeddings "{embeddings}" is not available or is not a valid path.')


    @property
    def embedding_length(self) -> int:
        return 1024

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        everything_embedded: bool = True
        chunk_size = 128

        for sentence in sentences:
            if self.name not in sentence._embeddings.keys(): everything_embedded = False

        if not everything_embedded:

            for idx, sentences_chunk in enumerate(chunks(sentences, chunk_size)):
                # Embed
                print('embed', idx)
                sentence_embeddings = self.bc.encode([sentence.to_original_text() for sentence in sentences_chunk])
                print('finished')
                for sentence, sentence_embedding in zip(sentences, sentence_embeddings):
                    sentence.set_embedding(self.name, torch.tensor(sentence_embedding))

        return sentences


class GoogleUseEmbeddings(EmbeddingBaseClass, FlairDocumentEmbeddings):
    """
    Class to infer the Google Universal Sentence Encoder to flair sentences.
    """

    def __init__(self):

        super().__init__()

        # tensorflow session
        self.session = tf.Session()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.encoder = self._make_embed_fn("https://tfhub.dev/google/universal-sentence-encoder-large/2")
        self.__embedding_length: int = 512
        self.name = f"{self.__class__.__name__}"
        self.static_embeddings = True

    def _make_embed_fn(self, module):
        with tf.Graph().as_default():
            sentences = tf.placeholder(tf.string)
            embed = hub.Module(module)
            embeddings = embed(sentences)
            self.session = tf.train.MonitoredSession()
        return lambda x: self.session.run(embeddings, {sentences: x})

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        everything_embedded: bool = True
        chunk_size = 128

        for sentence in sentences:
            if self.name not in sentence._embeddings.keys(): everything_embedded = False

        if not everything_embedded:

            for sentences_chunk in chunks(sentences, chunk_size):
                sentence_embeddings = self.encoder([sent.to_original_text() for sent in sentences_chunk])

                for sentence, sentence_embedding in zip(sentences_chunk, sentence_embeddings):
                    sentence.set_embedding(self.name, torch.tensor(sentence_embedding))


class InferSentEmbeddings(EmbeddingBaseClass, FlairDocumentEmbeddings):
    """
    Class to infer the InferSent embeddings to flair sentences. cf.
    `here <https://github.com/facebookresearch/InferSent>`_
    """

    def __init__(self, version=1):
        super().__init__()

        self.version = version
        if version == 1:
            self.PATH_TO_W2V = os.path.join(NLP_MODELS_PATH, 'pretrained', 'word_embeddings', 'glove.840B.300d',
                                        'glove.840B.300d.txt')
        if version == 2:
            self.PATH_TO_W2V = os.path.join(NLP_MODELS_PATH, 'pretrained', 'word_embeddings', 'crawl-300d-2M',
                                        'crawl-300d-2M.vec')

        self.MODEL_PATH = os.path.join(NLP_MODELS_PATH, 'pretrained', 'word_embeddings', 'infersent%s' % version
                                       , 'infersent%s.pkl' % version)

        # Set up logger
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

        # Load InferSent model
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': version}

        self.model = InferSent(params_model)
        self.model.load_state_dict(torch.load(self.MODEL_PATH))
        self.model.set_w2v_path(self.PATH_TO_W2V)

        self._embedding_length: int = params_model['enc_lstm_dim']

        self.name = f"{self.__class__.__name__ }_v{self.version}"
        self.static_embeddings = True

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        everything_embedded: bool = True
        infersent_sentences = []

        for sentence in sentences:
            if self.name not in sentence._embeddings.keys(): everything_embedded = False

        if not everything_embedded:
            for sentence in sentences:
                infersent_sentences.append(sentence.to_tokenized_string())

            self.model.build_vocab(infersent_sentences, tokenize=False)
            self.model.update_vocab(infersent_sentences, tokenize=False)
            embeddings = self.model.encode(infersent_sentences, tokenize=False)

            for sentence, sentence_embedding in zip(sentences, embeddings):
                sentence.set_embedding(self.name, torch.tensor(sentence_embedding))


class SkipThoughtEmbeddings(EmbeddingBaseClass, FlairDocumentEmbeddings):
    """
    Example! Class to infer the WordEmbeddings to flair sentences. Actually flair doc embeddings could be used
    """

    # todo: must be implemented!

    def __init__(self):
        super().__init__()
        self._embedding_length: int = 0

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass


class GenSenEmbeddings(EmbeddingBaseClass, FlairDocumentEmbeddings):
    """
    Example! Class to infer the WordEmbeddings to flair sentences. Actually flair doc embeddings could be used
    """

    # todo: must be implemented!

    def __init__(self):
        super().__init__()
        self._embedding_length: int = 0

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass


if __name__ == '__main__':
    from mtc.core.sentence import Sentence
    from time import time
    t1 = time()
    #document_embedding = DocumentEmbeddings('InferSentEmbeddings', version=2)
    document_embedding = DocumentEmbeddings('GoogleUseEmbeddings')
    print(time()-t1)
    token_embedding = TokenEmbeddings('WordEmbeddings', 'en')
    print(time()-t1)
    #document_embedding = DocumentEmbeddings('DocumentPoolEmbeddings', [token_embedding])
    #document_embedding = DocumentEmbeddings('BagOfWordsEmbeddings', 'WordEmbeddings')
    #document_embedding = DocumentEmbeddings('GoogleUseEmbeddings')
    #document_embedding = DocumentEmbeddings('DocumentPoolEmbeddings', [token_embedding])

    a = Sentence('Hello, how are you?')
    document_embedding.embed_str(a)
    #token_embedding.embed_str(a)
    print(a.embedding)
    #print(a.tokens[1].embedding)
    print(LOADED_EMBEDDINGS.models.keys())
