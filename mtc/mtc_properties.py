# import nltk
# from nltk.corpus import stopwords

# nltk.download('stopwords')
# STOP_WORDS = stopwords.words('english')
from mtc.helpers.util import LoadedModels

STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'm', 'o', 'y', 'could', 'ma', 'might', 'must', 'need', 'would']

STOP_WORDS = STOP_WORDS + ['medication', 'patient', 'the', 'left', 'right', 'denies', 'tablet', 'time']  # TESTS

KFOLDS = 150
EPOCHS = 10
TASK_NAME = 'n2c2'
LOADED_EMBEDDINGS = LoadedModels()

# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# STOP_WORDS = stopwords.words('german')
