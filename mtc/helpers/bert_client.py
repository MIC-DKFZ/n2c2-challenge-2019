from bert_serving.client import BertClient

bc = BertClient()
encoder = bc.encode(['First do it', 'then do it right', 'then do it better'])

# For interpretation of the output, see https://github.com/hanxiao/bert-as-service#getting-elmo-like-contextual-word-embedding
print(encoder)
