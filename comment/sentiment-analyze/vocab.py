from dataset import load_all_data
from utils import tokenize, generate_vocab
from json import dumps, load

docs, _ = load_all_data()

tokens = set()

for doc in docs:
    tokenized = tokenize(doc)
    tokens.update([t[0] for t in tokenized])

tokens = list(tokens)
vocab = generate_vocab(tokens)

_vocab = dumps(vocab)
file = open('data/vocab.json', 'w')
file.write(_vocab)
file.close()
