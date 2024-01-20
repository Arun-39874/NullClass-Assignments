import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import librosa
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog



df = pd.read_csv("voice.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

joblib.dump(model, "voice_model.pkl")

def extract_features(audio_data, _):
    try:
        audio_data = audio_data.astype(np.float32)
        features = np.mean(librosa.feature.mfcc(y=audio_data, sr=_).T, axis=0)
    except Exception as e:
        print("Error extracting features:", str(e))
        return None
    return features

def predict_gender(model, input_type, file_path=None):
    if input_type == 'file':
        try:
            audio_data, _ = librosa.load(file_path, sr=None)
        except Exception as e:
            print("Error loading audio file:", str(e))
            return None
    elif input_type == 'live':
        print("Recording 5 seconds of audio. Speak now...")
        audio_data = sd.rec(int(5 * 44100), samplerate=44100, channels=1, dtype=np.int16)
        sd.wait()
        audio_data = audio_data.flatten()
    else:
        print("Invalid input type. Choose 'file' or 'live'.")
        return None
    
    features = extract_features(audio_data, 44100)
    
    if features is not None:
        features = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features)
        predicted_gender = label_encoder.inverse_transform(prediction)[0]
        return predicted_gender
    else:
        return None

def choose_input_type():
    selected_option = input_type_combobox.get()
    if selected_option == "File":
        file_path = filedialog.askopenfilename(title="Select an audio file", filetypes=[("Audio files", "*.wav")])
        if file_path:
            predicted_gender = predict_gender(model, 'file', file_path)
            result_label.config(text=f"Predicted Gender: {predicted_gender}")
    elif selected_option == "Live":
        predicted_gender = predict_gender(model, 'live')
        result_label.config(text=f"Predicted Gender: {predicted_gender}")

root = tk.Tk()
root.title("Voice Gender Detection")

input_type_label = tk.Label(root, text="Select input type:")
input_type_label.grid(row=0, column=0, pady=10)

input_type_combobox = ttk.Combobox(root, values=["File", "Live"])
input_type_combobox.grid(row=0, column=1, pady=10)

process_button = tk.Button(root, text="Process", command=choose_input_type)
process_button.grid(row=1, column=0, columnspan=2, pady=10)

result_label = tk.Label(root, text="")
result_label.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()
