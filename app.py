

from flask import Flask, render_template, request
import os
import logging
import numpy as np
import librosa
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import logging
logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)  # Increase Flask debug logging

def convert_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
    except Exception as e:
        app.logger.error(f"Error converting audio file: {e}")
        return False
    return True

def load_audio(file_path):
    try:
        audio_data, _ = librosa.load(file_path, sr=16000)
    except Exception as e:
        app.logger.error(f"Error loading audio file: {e}")
        return None
    return audio_data

def extract_mfcc(audio_data):
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
    except Exception as e:
        app.logger.error(f"Error extracting MFCC features: {e}")
        return None
    return mfccs_mean

def preprocess_audio(audio_dir):
    X = []
    y = []
    for label, sub_dir in enumerate(["females", "males"]):
        sub_dir_path = os.path.join(audio_dir, sub_dir)
        for file_name in os.listdir(sub_dir_path):
            file_path = os.path.join(sub_dir_path, file_name)
            if file_path.endswith('.m4a'):
                wav_file_path = os.path.splitext(file_path)[0] + '.wav'
                if convert_to_wav(file_path, wav_file_path):
                    audio_data = load_audio(wav_file_path)
                    if audio_data is not None:
                        mfccs = extract_mfcc(audio_data)
                        if mfccs is not None:
                            X.append(mfccs)
                            y.append(label)
    return np.array(X), np.array(y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    data_dir = request.form['data_directory']
    data_dir = os.path.join(os.getcwd(), data_dir)  # Use current working directory
    
    X_train_scaled = None
    X_test_scaled = None
    
    try:
        X, y = preprocess_audio(data_dir)
        
        selector = SelectKBest(score_func=f_classif, k=20)
        X_selected = selector.fit_transform(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = np.expand_dims(X_train_scaled, axis=1)
        X_test_scaled = np.expand_dims(X_test_scaled, axis=1)

        num_classes = 2
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        model = Sequential([
          LSTM(units=128, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True),
          Dropout(0.5),
          LSTM(units=128),
          Dropout(0.5),
          Dense(units=num_classes, activation='softmax')
        ])


        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test))
        model.save('model.keras')
    except Exception as e:
        app.logger.error(f"Error training or saving the model: {e}")
        return 'Error training or saving the model'

    return 'Model trained successfully! <a href="http://127.0.0.1:5000/file"><button>Upload Another File</button></a>'

from gender import predict_gender




@app.route('/file')
def result():
    # Handle file upload and display result
    return render_template('result.html')

@app.route('/findgender', methods=['POST'])
def gender():
    uploaded_file = request.files['file']
   
    result = predict_gender(uploaded_file)
    return result

if __name__ == '__main__':
    app.run(debug=True)