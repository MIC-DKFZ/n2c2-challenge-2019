import os
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as nafc
from nlpaug.util import Action

from mtc.settings import NLP_MODELS_PATH

os.environ["MODEL_DIR"] = os.path.join(NLP_MODELS_PATH, 'nlpaug/')


class NLPAUG():

    def __init__(self):
        self.aug = nafc.Sequential([
            #naw.BertAug(action=Action.INSERT),
            naw.BertAug(action=Action.SUBSTITUTE),
            #naw.GloVeAug(model_path=os.environ.get("MODEL_DIR") + 'glove.6B.50d.txt', action=Action.SUBSTITUTE), bad results
            #naw.WordNetAug(),
            #naw.RandomWordAug(), # Deletes randomly word
        ])

    def augment(self, texts):
        all_texts = []
        for idx, text in enumerate(texts):
            all_texts.append(self.aug.augment(text))
            if idx%10 == 0:
                print(idx)
        return all_texts


if __name__ == '__main__':
    text = 'The quick brown fox jumps over the lazy dog'
    #aug = naw.FasttextAug(os.environ["MODEL_DIR"] + 'wiki-news-300d-1M.vec')
    nlp_aug = NLPAUG()
    augmented_text = nlp_aug.augment([text])
    print(augmented_text)
