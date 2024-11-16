import os
import numpy as np
from feature_extraction import load_audio, mfcc

def extract_features_from_directory(directory, n_mfcc=12):
    features = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            print(f"Processing command: {label}")
            for file in os.listdir(label_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(label_path, file)
                    print(f"Extracting features from {file_path}")
                    signal, samplerate = load_audio(file_path)
                    mfcc_features = mfcc(signal, samplerate, n_mfcc=n_mfcc)
                    features.append(mfcc_features)
                    labels.append(label)
    return features, labels

if __name__ == "__main__":
    data_dir = "data"
    features, labels = extract_features_from_directory(data_dir)
    print(f"Extracted features from {len(features)} files.")
    print(f"Labels: {set(labels)}")
    np.save("features.npy", features)
    np.save("labels.npy", labels)
    print("Features and labels saved to 'features.npy' and 'labels.npy'")
