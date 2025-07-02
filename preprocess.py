from music21 import converter, instrument, note, chord
import os
import pickle

def get_notes_from_midi(folder_path):
    notes = []
    for file in os.listdir(folder_path):
        if file.endswith(".mid") or file.endswith(".midi"):
            midi = converter.parse(os.path.join(folder_path, file))
            parts = instrument.partitionByInstrument(midi)

            elements = parts.parts[0].recurse() if parts else midi.flat.notes
            for element in elements:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

if __name__ == "__main__":
    notes = get_notes_from_midi("midi_songs")
    with open("notes.pkl", "wb") as f:
        pickle.dump(notes, f)
    print(f"[✔] Extracted {len(notes)} notes and saved to notes.pkl")
