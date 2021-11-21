import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

vocab_size = 300000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

#dataframe
df = pd.read_csv("training data/train_8.A.csv")


contents = df["msg"][0:30000].astype(str)
labels = df["count"][0:30000]


tokenizer = Tokenizer(oov_token=oov_tok,num_words=vocab_size)
tokenizer.fit_on_texts(contents)

word_index = tokenizer.word_index

training_seqs = tokenizer.texts_to_sequences(contents)
padded = pad_sequences(training_seqs)
padded = np.array(padded)

print(padded.shape)

model = tf.keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    layers.Dense(48, activation='relu'),
    layers.Dense(24, activation='relu'),
    layers.Bidirectional(layers.LSTM(80,return_sequences=True)),
    layers.Bidirectional(layers.LSTM(40)),
    layers.Dense(len(labels), activation='sigmoid')
])
model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['acc'])


model.fit(padded,labels,epochs=500,verbose=1)

model.summary()
model.save("model7.h5")
model.evaluate(padded,labels)