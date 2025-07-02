# train.py

import pickle
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation # type: ignore

def prepare_sequences(notes, seq_len=100):
    pitchnames = sorted(set(notes))
    note_to_int = {note: num for num, note in enumerate(pitchnames)}

    network_input, network_output = [], []
    for i in range(len(notes) - seq_len):
        seq_in = notes[i:i + seq_len]
        seq_out = notes[i + seq_len]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, seq_len, 1)) / float(len(pitchnames))
    network_output = np.array(network_output)

    return network_input, network_output, note_to_int, pitchnames

def create_model(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim))
    model.add(Activation("softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model

def train_model(notes, seq_len=100):
    network_input, network_output, note_to_int, pitchnames = prepare_sequences(notes, seq_len)
    model = create_model((network_input.shape[1], network_input.shape[2]), len(pitchnames))
    model.fit(network_input, network_output, epochs=50, batch_size=64)
    model.save("music_gen_model.h5")
    return model, network_input, pitchnames, note_to_int

# Optional: allows training directly if file is run standalone
if __name__ == "__main__":
    with open("notes.pkl", "rb") as f:
        notes = pickle.load(f)
    model, _, _, _ = train_model(notes)
    print("[✔] Model trained and saved as music_gen_model.h5")
