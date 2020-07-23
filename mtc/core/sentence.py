"""
.. module:: sentence
   :synopsis: Extending Sentence class of Flair!
.. moduleauthor:: Klaus Kades
"""

from typing import List, Dict, Union

from flair.data import Sentence as FlairSentence, Label
import torch


class Sentence(FlairSentence):
    """
    This class extends the Sentence class from Flair and can be used for preprocessing. Any parameters can be saved
    in the sentence_properties dictionary
    """

    def __init__(self,
                 raw_text: str,
                 preprocessor=None,
                 sentence_properties: Union[Dict, List[Dict]] = None,
                 use_tokenizer: bool = False,
                 labels: Union[List[Label], List[str]] = None,
                 ):
        self.sentence_properties = dict()

        if sentence_properties is not None:
            if type(sentence_properties) is dict:
                self.add_sentence_property(sentence_properties)
            else:
                self.add_sentence_properties(sentence_properties)

        if preprocessor is not None:
            text_for_flair = preprocessor.preprocess(raw_text)
        else:
            text_for_flair = raw_text

        super().__init__(text_for_flair, use_tokenizer=use_tokenizer, labels=labels)

    def to_string_tokens(self):
        return self.to_tokenized_string().split()

    def add_sentence_properties(self, sentence_properties):
        for sentence_property in sentence_properties:
            self.sentence_properties.update(sentence_property)

    def add_sentence_property(self, sentence_property):
        self.sentence_properties.update(sentence_property)

    def get_sentence_property(self, property_key):
        return self.sentence_properties[property_key]

    def get_embedding_by_name(self, embedding_name) -> torch.tensor:
        embedding = self._embeddings[embedding_name]
        return embedding
