import numpy as np
from hmm_model import HMM
from sklearn.cluster import KMeans
from joblib import dump

def normalize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-8  # Add epsilon to avoid division by zero
    return (features - mean) / std


def prepare_sequences(features, n_clusters=20):
    # Flatten all features for clustering
    all_features = np.vstack([normalize_features(f) for f in features])

    # Perform k-means clustering to quantize features into discrete indices
    print("Quantizing features into discrete observations...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_features)

    # Debugging for KMeans clustering
    print(f"KMeans cluster centers:\n{kmeans.cluster_centers_}")
    print(f"KMeans labels sample: {kmeans.labels_[:10]}")

    sequences = []
    for feature_set in features:
        sequence = kmeans.predict(normalize_features(feature_set))
        sequences.append(sequence)

    return sequences, kmeans


if __name__ == "__main__":
    # Load features and labels
    features = np.load("features.npy", allow_pickle=True)
    labels = np.load("labels.npy", allow_pickle=True)

    print(f"Features shape: {features.shape}")
    print(f"Features sample: {features[0]}")

    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError("Features contain NaN or Inf values. Check the feature extraction process.")


    # Prepare sequences for HMM training
    sequences, kmeans = prepare_sequences(features, n_clusters=20)

    # Save the KMeans model
    dump(kmeans, "kmeans_model.joblib")
    print("KMeans model saved to 'kmeans_model.joblib'")

    # Initialize HMM
    n_states = 7  # Increased number of states
    n_obs = 20  # Match with n_clusters
    hmm = HMM(n_states=n_states, n_obs=n_obs)

    # Train the HMM
    print("Training HMM...")
    hmm.train(sequences, max_iter=50)

    # Save the model parameters
    np.save("start_prob.npy", hmm.start_prob)
    np.save("trans_prob.npy", hmm.trans_prob)
    np.save("emit_prob.npy", hmm.emit_prob)
    print("HMM training completed and parameters saved.")
