from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

def lstm(input_length):
    model = Sequential()
    model.add(Embedding(input_dim = 188, output_dim = 50, input_length = input_length))
    model.add(LSTM(256, activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['mae', 'mse', 'accuracy'])
    return model
