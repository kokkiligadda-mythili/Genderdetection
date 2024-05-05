import os
import numpy as np
import librosa
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from pydub import AudioSegment

def convert_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        return f"Error converting audio file: {e}"

def load_audio(file_path):
    try:
        audio_data, _ = librosa.load(file_path, sr=16000)
        return audio_data
    except Exception as e:
        return f"Error loading audio file: {e}"

def extract_mfcc(audio_data):
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        return f"Error extracting MFCC features: {e}"

def predict_gender(uploaded_file):
    try:
        uploaded_file_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(uploaded_file_path)

        converted_wav_file_path = os.path.splitext(uploaded_file_path)[0] + '.wav'
        if not convert_to_wav(uploaded_file_path, converted_wav_file_path):
            return 'Error converting audio file'

        audio_data = load_audio(converted_wav_file_path)
        if audio_data is None:
            return 'Error loading audio file'

        mfccs = extract_mfcc(audio_data)
        if mfccs is None:
            return 'Error extracting MFCC features'

        model = load_model('model.keras')

        mfccs_transformed = mfccs.reshape(1, -1)
        selector = SelectKBest(score_func=f_classif, k=20)
        
        dummy_labels = np.zeros(1) 
        
        selected_mfccs = selector.fit_transform(mfccs_transformed, dummy_labels)

        scaler = StandardScaler()
        scaled_mfccs = scaler.fit_transform(selected_mfccs)

        input_data = np.expand_dims(scaled_mfccs, axis=0)

        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
        
        labels = ['female', 'male'] 
        gender = labels[predicted_class]

        return f"The predicted gender for {uploaded_file.filename} is: {gender}"
    except Exception as e:
        return f"Error predicting gender: {e}"
