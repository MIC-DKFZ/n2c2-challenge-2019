import pandas as pd
import random
from segtok.tokenizer import word_tokenizer
from segtok.tokenizer import split_contractions
# from googletrans import Translator
# #translator = Translator()
#
# translator = Translator(service_urls=[
#       'translate.google.com',
#       'translate.google.co.kr',
#     ])

from translate import Translator


def augment_sentences(raw_sentences, language_list=None, numb_translations=5, original_lang='en'):
    if language_list is None:
        # language_list = [key for key, item in googletrans.LANGUAGES.items()]
        language_list = ['fr', 'de', 'en', 'es', 'it', 'pt', 'ru', 'pl', 'no']
        language_list.remove(original_lang)
    if numb_translations is None:
        numb_translations = 5
    start = 0
    end = len(language_list) - 1

    translation_list = random.sample(range(start, end), numb_translations)

    translator_list = []
    translator_list.append(Translator(from_lang=original_lang, to_lang=language_list[translation_list[0]]))
    for tra_src, tra_dest in zip(translation_list[:-1], translation_list[1:]):
        translator_list.append(Translator(from_lang=language_list[tra_src], to_lang=language_list[tra_dest]))
    translator_list.append( Translator(from_lang=language_list[translation_list[-1]], to_lang=original_lang))

    new_raw_sentences = []
    for idx, original_text in enumerate(raw_sentences):
        translation_list = random.sample(range(start, end), numb_translations)

        trans_text = original_text
        for translator in translator_list:
            trans_text = translator.translate(trans_text)

        original_splitted = split_contractions(word_tokenizer(original_text))
        trans_splitted = split_contractions(word_tokenizer(trans_text))
        if original_splitted != trans_splitted:
            new_raw_sentences.append(trans_text)

        print(idx)
    return new_raw_sentences

def augment_sentences_backup(raw_sentences, language_list=None, numb_translations=5, original_lang='en'):
    if language_list is None:
        # language_list = [key for key, item in googletrans.LANGUAGES.items()]
        language_list = ['fr', 'de', 'en', 'es', 'it', 'pt', 'ru', 'pl', 'no']
        language_list.remove(original_lang)
    if numb_translations is None:
        numb_translations = 5
    start = 0
    end = len(language_list) - 1

    new_raw_sentences = []
    for idx, original_text in enumerate(raw_sentences):


        translation_list = random.sample(range(start, end), numb_translations)

        trans_text = original_text

        translator = Translator(from_lang=original_lang, to_lang=language_list[translation_list[0]])
        trans_text = translator.translate(trans_text)

        for tra_src, tra_dest in zip(translation_list[:-1], translation_list[1:]):
            translator = Translator(from_lang=language_list[tra_src], to_lang=language_list[tra_dest])
            trans_text = translator.translate(trans_text)
            #print(trans_text)
        translator = Translator(from_lang=language_list[translation_list[-1]], to_lang=original_lang)
        trans_text = translator.translate(trans_text)
        original_splitted = split_contractions(word_tokenizer(original_text))
        trans_splitted = split_contractions(word_tokenizer(trans_text))
        if original_splitted != trans_splitted:
            new_raw_sentences.append(trans_text)

        print(idx)
    return new_raw_sentences



if __name__ == '__main__':
    EXAMPLE_SENTENCES = [
        'terminal 1 is connected to the negative battery terminal',
        'there is no gap between terminal 6 and the positive terminal',
        'bulb a is still contained in a closed path with the battery .',
        'each bulb is in its own path',
        'a non-zero voltage means that the terminals are not connected .',
        'bulb a was still contained in the same closed path with the battery .',
        'bulb a was still contained in the same closed path with the battery .',
        'terminals 1 , 2 and 3 are separated from the positive battery terminal by a gap',
        'terminal 6 is separated by a gap from the negative battery terminal',
        'if a bulb and a switch are in the same path the switch affects the bulb',
        'a , b and c are in different paths',
        'a battery uses a chemical reaction to maintain different electrical states at the terminals',
        'the terminals are not connected',
        'bulb b is in a separate path',
        'bulb a is still contained in a closed path with the battery and switch z .',
        'bulb a was still contained in the same closed path with the battery .',
        'bulb a is still in a closed path with the battery',
        'there is no gap between terminal 6 and the positive terminal',
        'bulb c was not in a closed path',
        'a battery uses a chemical reaction to maintain different electrical states at the terminals',
        'terminals 1 , 2 and 3 are separated from the positive battery terminal by a gap',
        'the open switch creates a gap',
        'a and c are in the same closed path',
    ]

    EXAMPLE_SENTENCES = [
        'Kein Nachweis pathologisch vergrößerter Lymphknoten mediastinal und axillär beidseits.'
    ]
    new_sentences = augment_sentences(EXAMPLE_SENTENCES, original_lang='de')

    print(new_sentences)
