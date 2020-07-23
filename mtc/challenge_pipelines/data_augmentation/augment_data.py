import os
from mtc.helpers.nlpaug_fun import NLPAUG
from mtc.helpers.data_augmentation import augment_sentences
from mtc.helpers.file_management import save_augmented_sts_data, load_sts_data
from mtc.settings import NLP_MODELS_PATH, NLP_RAW_DATA

sts_data = load_sts_data(os.path.join('n2c2', 'clinicalSTS2019.train.txt'))

number_data = len(sts_data['raw_sentences_a'])
raw_sentences_a = sts_data['raw_sentences_a'][0:number_data]
raw_sentences_b = sts_data['raw_sentences_b'][0:number_data]
scores = sts_data['similarity_score'][0:number_data]

# numb_translations = 5
# new_raw_sentences_a = augment_sentences(raw_sentences_a, numb_translations=numb_translations)
# new_raw_sentences_b = augment_sentences(raw_sentences_b, numb_translations=numb_translations)
# new_scores = scores

nlp_aug = NLPAUG()
new_raw_sentences_a = nlp_aug.augment(raw_sentences_a)
new_raw_sentences_b = nlp_aug.augment(raw_sentences_b)
new_scores = scores

sts_data['raw_sentences_a'] = sts_data['raw_sentences_a'] + new_raw_sentences_a
sts_data['raw_sentences_b'] = sts_data['raw_sentences_b'] + new_raw_sentences_b
sts_data['similarity_score'] = sts_data['similarity_score'] + new_scores

print('#####')
print(raw_sentences_a)
print(new_raw_sentences_a)
print('#####')
print(raw_sentences_b)
print(new_raw_sentences_b)
save_augmented_sts_data(sts_data, os.path.join('n2c2', 'clinicalSTS2019.augmented.train.txt'))
