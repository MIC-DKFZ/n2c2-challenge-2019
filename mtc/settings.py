import os
from dotenv import load_dotenv
from pathlib import Path

repo_root = Path(__file__).parent.parent
load_dotenv(repo_root / '.env')

# Path to the pretrained models
NLP_MODELS_PATH = os.environ.get('NLP_MODELS_PATH')

# Path used to store results from experiments
NLP_EXPERIMENT_PATH = os.environ.get('NLP_EXPERIMENT_PATH')

#: Path to the folder with the training data.
#: There should be an additional folder n2c2 inside so that the training data is located at
#: ``NLP_RAW_DATA/n2c2/clinicalSTS2019.train.txt``.
NLP_RAW_DATA = os.environ.get('NLP_RAW_DATA')

if 'PATH_TO_INFERSENT' in os.environ:
    PATH_TO_INFERSENT = os.environ.get('PATH_TO_INFERSENT')

if 'ZEROMQ_SOCK_TMP_DIR' in os.environ:
    ZEROMQ_SOCK_TMP_DIR = os.environ.get('ZEROMQ_SOCK_TMP_DIR')

if 'PATH_TO_SENTEVAL' in os.environ:
    PATH_TO_SENTEVAL = os.environ.get('PATH_TO_SENTEVAL')

if 'PATH_TO_SENT_EVAL_DATA' in os.environ:
    PATH_TO_SENT_EVAL_DATA = os.environ.get('PATH_TO_SENT_EVAL_DATA')

# https://sbmi.uth.edu/ccb/resources/medex.htm
if 'MEDEX_PATH' in os.environ:
    MEDEX_PATH = os.environ.get('MEDEX_PATH')
