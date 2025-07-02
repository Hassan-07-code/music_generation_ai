# generate.py

import pickle
import numpy as np
import random
from music21 import stream, note, chord, instrument
from tensorflow.keras.models import load_model  # type: ignore

def generate_music(model, network_input, pitchnames, note_to_int, n_notes=500, seq_len=100):
    """
    Generate new music using a trained model and save as a MIDI file.
    """
    int_to_note = {num: note for note, num in note_to_int.items()}

    # Pick a random seed and reshape to 3D (1, seq_len, 1)
    start = np.random.randint(0, len(network_input) - 1)
    pattern = np.reshape(network_input[start], (1, seq_len, 1))

    prediction_output = []

    for _ in range(n_notes):
        # Predict the next note
        prediction = model.predict(pattern, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        # Normalize the new note and reshape for LSTM input
        new_note = np.array([[[index / float(len(pitchnames))]]])
        pattern = np.concatenate((pattern, new_note), axis=1)
        pattern = pattern[:, 1:, :]  # keep only last seq_len steps

    # Convert note predictions to music21 objects
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            # It's a chord
            chord_notes = [note.Note(int(n)) for n in pattern.split('.')]
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            # It's a single note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5  # quarter note step

    # Create stream and write to MIDI file
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')
    print("[✔] MIDI file saved: output.mid")

# Optional: standalone run (for testing)
if __name__ == "__main__":
    with open("notes.pkl", "rb") as f:
        notes = pickle.load(f)

    model = load_model("music_gen_model.h5")

    from train import prepare_sequences
    input_seq, _, note_to_int, pitchnames = prepare_sequences(notes)
    generate_music(model, input_seq, pitchnames, note_to_int)
