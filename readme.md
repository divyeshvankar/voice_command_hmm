
---
# Voice-Controlled Quiz System

### **Author**: Vankar Divyesh Kumar  
### **Roll Number**: 210101108  
### **Year**: B.Tech, 4th Year  

---

## **Overview**

This project implements a **real-time quiz system** navigated by **voice commands** using **Hidden Markov Models (HMMs)** for speech recognition. The system recognizes three commands: `start`, `next`, and `stop`, which control the quiz's flow. 

### **Key Features**
1. **Voice Command Recognition**:
   - Extracts **MFCC features** from audio input.
   - Uses **KMeans clustering** for feature quantization.
   - Predicts commands using a pre-trained **HMM**.

2. **Quiz System Integration**:
   - Supports commands:
     - **`start`**: Starts the quiz.
     - **`next`**: Moves to the next question.
     - **`stop`**: Ends the quiz.
   - Displays questions and captures answers interactively.

---

## **File Structure**
```plaintext
.
├── requirements.txt          # Dependencies
├── readme.md                 # Documentation
├── .gitignore                # Ignore unnecessary files
├── training/                 # Training scripts and models
│   ├── feature_extraction.py # MFCC feature extraction
│   ├── hmm_model.py          # HMM implementation
│   ├── start_prob.npy        # Start probabilities
│   ├── trans_prob.npy        # Transition probabilities
│   ├── emit_prob.npy         # Emission probabilities
│   ├── kmeans_model.joblib   # Pre-trained KMeans model
├── realtime/                 # Real-time quiz scripts
│   ├── quiz_session.py       # Main script for quiz and recognition
│   ├── quiz_questions.py     # Quiz questions and flow logic
│   ├── live_command.wav      # Temporary file for live recording
```

---

## **Setup and Installation**

### **Step 1: Clone the Repository**
```bash
git clone <repository-url>
cd <repository-directory>
```

### **Step 2: Set Up a Virtual Environment (Optional)**
```bash
python -m venv myenv
source myenv/bin/activate    # For Linux/MacOS
myenv\Scripts\activate       # For Windows
```

### **Step 3: Install Dependencies**
Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## **How to Run**

1. **Navigate to the `realtime` Directory**:
   ```bash
   cd realtime
   ```

2. **Start the Quiz Session**:
   ```bash
   python quiz_session.py
   ```

3. **Use Voice Commands**:
   - Speak one of the following commands:
     - `start` to begin the quiz.
     - `next` to proceed to the next question.
     - `stop` to end the quiz.

---

## **Voice Command Functionality**
The quiz system recognizes the following commands:
- **`start`**:
  - Begins the quiz by showing the first question.
- **`next`**:
  - Displays the next question.
- **`stop`**:
  - Ends the quiz session.

---

## **Notes**
1. Ensure that:
   - A **microphone** is connected for recording.
   - The environment is quiet for optimal recognition.
2. All trained files (`start_prob.npy`, `trans_prob.npy`, `emit_prob.npy`, `kmeans_model.joblib`) should be present in the `training` directory.

---

## **How It Works**

1. **Training (Pre-computed Models)**:
   - **Feature Extraction**:
     - Extracts **MFCC features** from pre-recorded audio samples.
   - **Clustering**:
     - Uses **KMeans** to cluster features into discrete observations.
   - **HMM Training**:
     - Trains an **HMM** using these observations to recognize `start`, `next`, and `stop`.

2. **Real-Time Recognition**:
   - **Audio Recording**:
     - Records 3 seconds of live audio input.
   - **Feature Quantization**:
     - Processes the audio into discrete observations using the trained **KMeans model**.
   - **Command Prediction**:
     - Passes the observations to the **HMM** for command prediction.

3. **Quiz Integration**:
   - Displays and navigates quiz questions based on the recognized commands.

---

## **Dependencies**
The following dependencies are required (listed in `requirements.txt`):
- `numpy`
- `scipy`
- `sounddevice`
- `wave`
- `joblib`
- `sklearn`

---

## **Acknowledgments**
- Hidden Markov Models for speech recognition.
- KMeans clustering for feature quantization.
- Python libraries for audio processing and model implementation.

---

