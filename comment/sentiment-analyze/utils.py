from konlpy.tag import Okt
from json import load

pos_tagger = Okt()

def tokenize(doc):
    return [(t[0], t[1]) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def generate_vocab(tokens):
    vocab = {
        '#UNKNOWN': 0,
        '#PADDING': 1
    }

    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab

def get_token_index(token, vocab):
    if token in vocab:
        return vocab[token]
    else:
        return vocab['#UNKNOWN']

def load_vocab():
    with open('vocab.json', 'r') as file:
        vocab = load(file)

    return vocab

def pad_row(row):
    pad_len = max(0, 60 - len(row))
    row = row + [1] * pad_len

    return row
