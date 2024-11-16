
---

# Real-Time Voice Command Quiz System

**Developed by:**  
**Vankar Divyesh Kumar**  
**Roll Number:** 210101108  
**Email :** k.vankar@iitg.ac.in  
**B.Tech 4th Year**

---

This project is a real-time voice command-based quiz system implemented using Python. It utilizes Hidden Markov Models (HMM) for recognizing commands like `start`, `next`, and `stop` and integrates a quiz system to conduct interactive quizzes.

## Features
1. **Voice Command Recognition**: Supports `start`, `next`, and `stop` commands.
2. **Interactive Quiz System**: Dynamically fetches questions and processes user responses based on voice commands.
3. **Real-time Processing**: Records, processes, and predicts commands on-the-fly.

---

## File Structure

### Training Module
1. **`feature_extraction.py`**  
   Handles the extraction of MFCC features from audio files.

2. **`hmm_model.py`**  
   Implements the Hidden Markov Model (HMM) class and training methods.

3. **`train_hmm.py`**  
   Script to train the HMM model using pre-recorded training data.

4. **Data**:
   - Pre-trained model files:
     - `start_prob.npy`
     - `trans_prob.npy`
     - `emit_prob.npy`
     - `kmeans_model.joblib`

---

### Real-Time Module
1. **`realtime/quiz_session.py`**  
   Main script to run the real-time voice command recognition and integrate it with the quiz system.

2. **`realtime/questions.json`**  
   Contains the quiz questions in JSON format.

3. **Utility Functions**:  
   - `record_live_audio()`: Records live audio input.
   - `predict_command()`: Predicts commands using the trained HMM model.

---

## Prerequisites

1. **Python Environment**
   - Python 3.7 or above.
   - Create a virtual environment:  
     ```bash
     python -m venv myenv
     ```
   - Activate the environment:
     - Windows: `myenv\Scripts\activate`
     - Linux/Mac: `source myenv/bin/activate`

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   **Dependencies include**:
   - `numpy`
   - `sounddevice`
   - `scikit-learn`
   - `joblib`
   - `wave`
   - `json`

3. **Recording Hardware**
   - Ensure a working microphone is connected.

---

## How to Run

### Training the Model (Optional)
1. Place your training audio files in a folder structure as follows:
   ```
   training_data/
   ├── start/
   ├── next/
   ├── stop/
   ```
2. Extract MFCC features and train the HMM:
   ```bash
   python train_hmm.py
   ```
3. Ensure the trained files (`*.npy` and `kmeans_model.joblib`) are in the `training/` folder.

---

### Running the Quiz System
1. Navigate to the `realtime/` directory:
   ```bash
   cd realtime
   ```
2. Start the quiz session:
   ```bash
   python quiz_session.py
   ```
3. Follow the on-screen instructions:
   - Speak commands like `start`, `next`, or `stop`.
   - The system will fetch quiz questions, present them, and wait for your response using voice commands.

---

## Additional Notes

1. **Testing and Debugging**
   - Test individual commands by recording and predicting using `quiz_session.py`.
   - Use `print` statements to debug any unexpected behavior.

2. **Expanding the Quiz System**
   - Add more questions in the `questions.json` file.
   - Ensure that question IDs are unique and responses are well-defined.

3. **Voice Training Data**
   - For better accuracy, include a variety of recordings for each command during training.

---

For further assistance or issues, contact **Vankar Divyesh Kumar** at the provided project repository or email.

--- 
