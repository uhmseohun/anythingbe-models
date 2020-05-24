from utils\
    import tokenize, get_token_index, load_vocab, pad_row

vocab = load_vocab()

def load_file(file_path):
    with open(file_path, 'r') as file:
        rows = file.readlines()[1:]

    rows = [row.strip().split('\t') for row in rows]
    
    x_data = [row[1] for row in rows]

    for index, row in enumerate(x_data):
        row = tokenize(row)
        row = [r[0] for r in row]
        row = [get_token_index(r, vocab) for r in row]
        row = pad_row(row)
        x_data[index] = row

    y_data = [row[2] for row in rows]

    return x_data, y_data

def load_train_data():
    return load_file('data/ratings_train.txt')

def load_test_data():
    return load_file('data/ratings_test.txt')

def load_all_data():
    return load_file('data/ratings.txt')
