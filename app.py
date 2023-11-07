import gradio as gr
import soundfile as sf
import tempfile
import shutil
import os
import librosa
import time
import numpy as np
import subprocess 
from pywhispercpp.model import Model

# model = Model('base.en', n_threads=6,models_dir="./Models") # Only english
# model = Model('base', n_threads=6,models_dir="./Models",language="hindi",translate=False)  # Multilingual
model = Model('medium', n_threads=6,models_dir="./Models",language="hindi",translate=False)  # Multilingual

def resample_to_16k(audio, orig_sr):
    y_resampled = librosa.resample(y=audio, orig_sr=orig_sr, target_sr = 16000)
    return y_resampled

def transcribe(audio):
    print(type(audio))
    sr,y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    y_resampled = resample_to_16k(y, sr)
    
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
        sf.write(temp_audio_path, y_resampled, 16000)

    start_time_py = time.time()
    py_result = model.transcribe(f'{temp_audio_path}', n_threads=6)
    end_time_py = time.time()
    print("Py_result : ",py_result)
    print("--------------------------")
    print(f"Execution time using py: {end_time_py - start_time_py} seconds")
    output_text = ""
    for segment in py_result:
        output_text+=segment.text
    return output_text, (end_time_py - start_time_py)



demo = gr.Interface(
    transcribe,
    inputs = "microphone",
    # gr.Audio(sources=["microphone"]),
    outputs=[gr.Textbox(label="Py_Transcription"),gr.Textbox(label="Time taken for Transcription")],
    # examples=["./Samples/Hindi_1.mp3","./Samples/Hindi_2.mp3","./Samples/Tamil_1.mp3","./Samples/Tamil_2.mp3","./Samples/Marathi_1.mp3","./Samples/Marathi_2.mp3","./Samples/Nepal_1.mp3","./Samples/Nepal_2.mp3","./Samples/Telugu_1.wav","./Samples/Telugu_2.wav","./Samples/Malayalam_1.wav","./Samples/Malayalam_2.wav","./Samples/Gujarati_1.wav","./Samples/Gujarati_2.wav","./Samples/Bengali_1.wav","./Samples/Bengali_2.wav"]
    examples=["./Samples/Hindi_1.mp3","./Samples/Hindi_2.mp3","./Samples/Hindi_3.mp3","./Samples/Hindi_4.mp3","./Samples/Hindi_5.mp3"] # only hindi   # ,"./Samples/Tamil_1.mp3","./Samples/Tamil_2.mp3","./Samples/Marathi_1.mp3","./Samples/Marathi_2.mp3","./Samples/Nepal_1.mp3","./Samples/Nepal_2.mp3","./Samples/Telugu_1.wav","./Samples/Telugu_2.wav","./Samples/Malayalam_1.wav","./Samples/Malayalam_2.wav","./Samples/Gujarati_1.wav","./Samples/Gujarati_2.wav","./Samples/Bengali_1.wav","./Samples/Bengali_2.wav"]
)

demo.launch()