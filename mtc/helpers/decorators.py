from typing import Union, List
import functools

from mtc.core.sentence import Sentence
import time


def convert_sentence_to_list(func):
    """
    If only one sentence is passed, convert to list of sentence
    """
    @functools.wraps(func)
    def wrapper(embedding_instance, sentences: Union[Sentence, List[Sentence]], *args, **kwargs):
        if type(sentences) is Sentence:
            sentences = [sentences]
        return func(embedding_instance, sentences, *args, **kwargs)
    return wrapper


def timeit(func):
    """
    Measures the time of a function call.
    """
    @functools.wraps(func)
    def wrapper(class_instance, *args, **kwargs):
        ts = time.time()
        result = func(class_instance, *args, **kwargs)
        te = time.time()
        class_instance.time_log.append({func.__name__: te-ts})
        return result
    return wrapper