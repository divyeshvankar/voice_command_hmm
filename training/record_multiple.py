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

def record_multiple_commands(command_name, num_samples, duration=3):
    base_dir = f"data/{command_name}/"
    for i in range(num_samples):
        filename = f"{base_dir}{command_name}_{i + 1}.wav"
        print(f"\nRecording sample {i + 1}/{num_samples} for command '{command_name}'")
        record_audio(filename, duration=duration)
        print(f"Sample {i + 1}/{num_samples} saved as {filename}")

if __name__ == "__main__":
    print("Voice Command Recording Script")
    print("Available commands: start, next, stop")
    command = input("Enter the command name to record: ").strip()
    num_samples = int(input("Enter the number of samples to record: "))
    duration = int(input("Enter the duration of each recording (in seconds): "))
    record_multiple_commands(command, num_samples, duration)
