import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

start_idx = 10001
end_idx = 10010

def get_content(review):
  return review['review']

def get_rating(review):
  return 1 if int(review['rating']) > 5 else 0

# Get all data
reviews = []
ratings = []
for idx in range(start_idx, end_idx+1):
  with open(f'./train-data/{idx}.json') as file:
    jsoned = json.load(file)
    reviews = reviews + list(map(get_content, jsoned))
    ratings = ratings + list(map(get_rating, jsoned))

# Vectorize sentences
t = Tokenizer()
t.fit_on_texts(reviews)
vocab_size = len(t.word_index) + 1

reviews = t.texts_to_sequences(reviews)
reviews = pad_sequences(reviews, padding='post')

# Tuplize
data = (reviews, ratings)
