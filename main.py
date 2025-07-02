# main.py — Fully Automated Music Generation
from preprocess import get_notes_from_midi
from train import train_model
from generate import generate_music

def main():
    print("🎶 Starting Music Generation AI Pipeline...")

    # Step 1: Preprocess MIDI files
    print("\n🔍 Preprocessing MIDI files...")
    notes = get_notes_from_midi("midi_songs")
    print(f"[✔] Extracted {len(notes)} notes.")

    # Step 2: Train the model
    print("\n🧠 Training the LSTM model...")
    model, network_input, pitchnames, note_to_int = train_model(notes)
    print("[✔] Model trained and saved as music_gen_model.h5")

    # Step 3: Generate music and save as MIDI
    print("\n🎼 Generating new music...")
    generate_music(model, network_input, pitchnames, note_to_int)
    print("[🎉] Music generated and saved as output.mid")

    print("\n✅ Pipeline complete! Enjoy your AI-generated melody 🎵")

if __name__ == "__main__":
    main()
