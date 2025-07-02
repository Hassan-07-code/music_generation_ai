# 🎼 Music Generation with AI

This project demonstrates how to use Deep Learning (LSTM) to generate original music compositions by training on MIDI files. Built using Python, TensorFlow, and the `music21` library, the system learns patterns in classical or jazz music and generates new melodies, saving them as playable MIDI files.

---

## 🚀 Features

- 🎹 Preprocess and extract notes from MIDI music data
- 🧠 Train an LSTM neural network to learn music patterns
- 🎶 Generate new, AI-composed music sequences
- 💾 Export generated music to a `.mid` file for playback
- 🎻 Supports chords, single notes, and piano instrumentation

---

## 📁 Project Structure

music_generation_ai/
├── midi_songs/ # Folder containing training MIDI files
├── notes.pkl # Saved notes for training and generation
├── music_gen_model.h5 # Trained LSTM model
├── output.mid # Generated music output
├── preprocess.py # MIDI parsing and preprocessing
├── train.py # LSTM model creation and training
├── generate.py # Generate and export new music
├── main.py # Auto-run pipeline: preprocess → train → generate
└── requirements.txt # Python dependencies


---

## 🔧 Setup Instructions

1. **Install dependencies:**

```bash
pip install -r requirements.txt
