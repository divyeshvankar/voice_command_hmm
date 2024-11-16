import numpy as np
import sounddevice as sd
import wave
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))
from feature_extraction import mfcc, load_audio
from hmm_model import HMM
from joblib import load 
import numpy as np
import os
from feature_extraction import mfcc, load_audio
from hmm_model import HMM
from joblib import load
from sounddevice import rec, wait
import wave

# Helper Functions
def record_live_audio(filename, duration=3, samplerate=44100):
    print(f"Recording for {duration} seconds...")
    audio_data = rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    wait()
    print("Recording completed.")
    audio_data = np.int16(audio_data * 32767)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    print(f"Audio saved to {filename}")

def predict_command(hmm, kmeans, filename):
    signal, samplerate = load_audio(filename)
    mfcc_features = mfcc(signal, samplerate)
    observations = kmeans.predict(mfcc_features)
    state_sequence = hmm.predict(observations)
    return state_sequence

# Quiz Data
quiz_data = [
    {"question": "What is the capital of France?", "options": ["Paris", "Berlin", "Rome"], "answer": "Paris"},
    {"question": "What is 2 + 2?", "options": ["3", "4", "5"], "answer": "4"},
    {"question": "What is the color of the sky?", "options": ["Blue", "Green", "Red"], "answer": "Blue"},
]

# Command-State Mapping
state_to_command = {
    0: "start",
    1: "next",
    2: "start",
    3: "stop",
    4: "unknown",
    5: "unknown",
    6: "stop"
}

# Quiz System
def run_quiz(hmm, kmeans):
    print("Starting Quiz...")
    question_index = 0
    quiz_running = True

    while quiz_running and question_index < len(quiz_data):
        question = quiz_data[question_index]
        print(f"\nQuestion {question_index + 1}: {question['question']}")
        for i, option in enumerate(question["options"], start=1):
            print(f"{i}. {option}")

        while True:
            print("\nSay a command (e.g., 'start', 'next', 'stop')...")
            record_live_audio("live_command.wav", duration=3)
            state_sequence = predict_command(hmm, kmeans, "live_command.wav")
            most_frequent_state = max(set(state_sequence), key=list(state_sequence).count)
            command = state_to_command.get(most_frequent_state, "unknown")
            print(f"Predicted command: {command}")

            if command == "next":
                question_index += 1
                break
            elif command == "stop":
                quiz_running = False
                print("Quiz stopped.")
                break
            elif command == "start":
                print("You already started the quiz!")
            else:
                print("Unknown command. Please try again.")

    if question_index >= len(quiz_data):
        print("\nQuiz completed! Thanks for participating.")
    else:
        print("\nQuiz exited early.")

# Main Script
if __name__ == "__main__":
    print("Real-Time Quiz System")
    training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training'))
    start_prob = np.load(os.path.join(training_dir, "start_prob.npy"))
    trans_prob = np.load(os.path.join(training_dir, "trans_prob.npy"))
    emit_prob = np.load(os.path.join(training_dir, "emit_prob.npy"))

    kmeans = load(os.path.join(training_dir, "kmeans_model.joblib"))
    n_states, n_obs = trans_prob.shape[0], emit_prob.shape[1]
    hmm = HMM(n_states=n_states, n_obs=n_obs)
    hmm.start_prob, hmm.trans_prob, hmm.emit_prob = start_prob, trans_prob, emit_prob

    while True:
        print("\nSay a command to start the quiz (e.g., 'start', 'next', 'stop')...")
        record_live_audio("live_command.wav", duration=3)
        state_sequence = predict_command(hmm, kmeans, "live_command.wav")
        most_frequent_state = max(set(state_sequence), key=list(state_sequence).count)
        command = state_to_command.get(most_frequent_state, "unknown")
        print(f"Predicted command: {command}")

        if command == "start":
            run_quiz(hmm, kmeans)
        elif command == "stop":
            print("Exiting the program.")
            break
        else:
            print("Unknown command. Say 'start' to begin or 'stop' to exit.")
