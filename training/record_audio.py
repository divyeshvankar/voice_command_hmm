import sounddevice as sd
import numpy as np
import wave
import os

def record_audio(filename, duration=3, samplerate=44100):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording completed. Saving to file.")
    audio_data = np.int16(audio_data * 32767)  # Convert to 16-bit PCM
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    print(f"Audio saved to {filename}")

# Test the function
if __name__ == "__main__":
    record_audio("data/start_command.wav", duration=3)
