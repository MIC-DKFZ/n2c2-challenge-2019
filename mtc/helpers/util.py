from typing import List, Dict, Union
import re
import json

import torch

class LoadedModels:
    """
    Class that caches the loaded models
    """

    @staticmethod
    def _get_key(name, *args, **kwargs):
        return f'{name}_{str(args)}_{str(kwargs)}'

    def __init__(self):

        self.models = dict()

    def get_class_instance(self, cls, name, *args, **kwargs):
        if not self._key_exists(name, *args, **kwargs):
            print('loading', self._get_key(name, *args, **kwargs))
            self._add_model(cls(*args, **kwargs), name, *args, **kwargs)
        return self._get_model(name, *args, **kwargs)

    def _key_exists(self, name, *args, **kwargs):
        if self._get_key(name, *args, **kwargs) not in self.models.keys():
            return False
        else:
            return True

    def _add_model(self, model_instance, name, *args, **kwargs):
        self.models.update({self._get_key(name, *args, **kwargs): model_instance})

    def _get_model(self, name, *args, **kwargs):
        return self.models[self._get_key(name, *args, **kwargs)]


class PipelineDictArgument(dict):

    def __init__(self, *args, **kwargs):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        super().__init__({
            'args': args,
            'kwargs': kwargs
        })

    def __str__(self):

        return json.dumps(self)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_token_embedding_by_names(token, token_names):
    """
    Adapted flair.data.Token get_embedding()
    :param token:
    :param token_names:
    :return:
    """
    token_embeddings = [
        token._embeddings[embed] for embed in sorted(token_names)
    ]
    if token_embeddings:
        return torch.cat(token_embeddings, dim=0)
    else:
        return torch.Tensor()


def get_med(sentence):
    match = re.search(r'\[([^]]+)\]', sentence)
    if match:
        return match.group(1)
    else:
        return ''


def get_ingredient_booleans(raw_sentences: List[str]):
    ingredient_booleans = []
    for raw_sentence in raw_sentences:
        match = re.search(r'\[([^]]+)\]', raw_sentence)
        ingredient_booleans.append(match)
    return ingredient_booleans

