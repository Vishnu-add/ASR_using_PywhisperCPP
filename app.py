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

model = Model('base.en', n_threads=6)

def resample_to_16k(audio, orig_sr):
    y_resampled = librosa.resample(y=audio, orig_sr=orig_sr, target_sr = 16000)
    return y_resampled

def transcribe(audio,):
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
    gr.Audio(sources=["microphone"]),
    outputs = [gr.Textbox(label="Py_Transcription"), gr.Textbox(label="Time taken for Transcription")]
)

demo.launch()
