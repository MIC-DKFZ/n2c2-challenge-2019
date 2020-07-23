import os
from mtc.helpers.util import PipelineDictArgument
from mtc.core.preprocessor import Preprocessor
from mtc.core.sentence import Sentence
from mtc.core.embedding_training import EmbeddingTraining

if __name__ == '__main__':

    from mtc.helpers.file_management import save_augmented_sts_data, load_sts_data

    pa_preprocessor = PipelineDictArgument('SelectivePreprocessor', [
        PipelineDictArgument('NumberUnifier'),
        PipelineDictArgument('SpellingCorrector'),
        PipelineDictArgument('SentenceTokenizer'),
        PipelineDictArgument('WordTokenizer'),
        PipelineDictArgument('PunctuationRemover'),
        PipelineDictArgument('StopWordsRemover'),
        PipelineDictArgument('LowerCaseTransformer'),
        PipelineDictArgument('Lemmatizer'),
    ])

    sts_data = load_sts_data(os.path.join('n2c2', 'clinicalSTS2019.train.txt'))

    number_data = len(sts_data['raw_sentences_a'])
    raw_sentences_a = sts_data['raw_sentences_a']
    raw_sentences_b = sts_data['raw_sentences_b']

    preprocessor = Preprocessor(*pa_preprocessor['args'], **pa_preprocessor['kwargs'])

    preprocessed_texts = preprocessor.preprocess(raw_sentences_a + raw_sentences_b)
    sentences_combined = [Sentence(sen) for sen in preprocessed_texts if sen != '']

    embedding_training = EmbeddingTraining('Doc2VecTraining', 'doc2vec_trained', train_dict={'epochs': 10}, build_dict={'min_count': 1})
    embedding_training.train(sentences_combined)
    embedding_training.save_model()
