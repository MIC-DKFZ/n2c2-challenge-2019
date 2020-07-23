import os

from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser

from mtc.settings import NLP_MODELS_PATH, ZEROMQ_SOCK_TMP_DIR

os.environ["ZEROMQ_SOCK_TMP_DIR"] = ZEROMQ_SOCK_TMP_DIR
model_dir = os.path.join(NLP_MODELS_PATH, 'pretrained', 'word_embeddings', 'bert_models', 'uncased_L-24_H-1024_A-16')

args = get_args_parser().parse_args(['-model_dir', model_dir,
                                     '-port', '5555',
                                     '-port_out', '5556',
                                     '-pooling_strategy', 'CLS_TOKEN',
                                     '-max_seq_len', '128',
                                     #'-graph_tmp_dir', '/home/klaus/private_klaus/bert_trash',
                                     '-show_tokens_to_client',
                                     '-mask_cls_sep'])
server = BertServer(args)
server.start()
