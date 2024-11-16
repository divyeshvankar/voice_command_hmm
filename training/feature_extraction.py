import numpy as np
import wave
from scipy.fftpack import dct

def load_audio(filename):
    with wave.open(filename, 'r') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        audio_signal = np.frombuffer(audio_data, dtype=np.int16)
    return audio_signal, framerate

def mfcc(signal, samplerate, n_mfcc=12, n_fft=512, n_filters=40):
    # Step 1: Pre-emphasis
    emphasized_signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    # Step 2: Frame signal
    frame_size = 0.025
    frame_stride = 0.01
    frame_length = int(round(frame_size * samplerate))
    frame_step = int(round(frame_stride * samplerate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    padded_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)
    ).T
    frames = padded_signal[indices.astype(np.int32, copy=False)]

    # Step 3: Apply Hamming window
    frames *= np.hamming(frame_length)

    # Step 4: Fourier-Transform and Power Spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)

    # Step 5: Filter Banks
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (samplerate / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin = np.floor((n_fft + 1) * hz_points / samplerate)

    fbank = np.zeros((n_filters, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_filters + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # Step 6: Apply Discrete Cosine Transform
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    return mfcc

# Test the feature extraction
if __name__ == "__main__":
    # Load the recorded audio
    audio_signal, samplerate = load_audio("data/start_command.wav")
    # Extract MFCC features
    mfcc_features = mfcc(audio_signal, samplerate)
    print("MFCC Features Shape:", mfcc_features.shape)
