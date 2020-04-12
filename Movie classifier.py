# comment these 2 lines out if not running on mac
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.datasets import imdb
from keras.preprocessing import sequence

VOCAB_SIZE = 5000
INDEX_FROM = 3
MAX_SEQ_LEN = 128
EMBEDDING_DIM = 64

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE, index_from=INDEX_FROM)

word_to_idx = imdb.get_word_index()
idx_to_word = {v+INDEX_FROM: k for k,v in word_to_idx.items()}
idx_to_word[0] = '<PAD>'
idx_to_word[1] = '<START>'
idx_to_word[2] = '<UNK>'

X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQ_LEN)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQ_LEN)

model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQ_LEN))

model.add(LSTM(128, return_sequences=True)) 
model.add(LSTM(128))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

tensorboard = keras.callbacks.TensorBoard(log_dir='sentiment_logs')

model.fit(X_train, y_train, epochs=20, batch_size=256, validation_split=0.2, callbacks=[tensorboard])

score = model.evaluate(X_test, y_test)
print('Accuracy: {0:.4f}'.format(score[1]))


# In[ ]:




