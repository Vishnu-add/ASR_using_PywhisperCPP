# ASR_using_pywhispercpp

Run
```
pip install -r requirements.txt
```

# pywhispercpp
Python bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with a simple Pythonic API on top of it.


# Installation 

1. Install [ffmpeg](https://ffmpeg.org/)

 ```bash
 # on Ubuntu or Debian
 sudo apt update && sudo apt install ffmpeg

 # on Arch Linux
sudo pacman -S ffmpeg

 # on MacOS using Homebrew (https://brew.sh/)
 brew install ffmpeg

 # on Windows using Chocolatey (https://chocolatey.org/)
 choco install ffmpeg

 # on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

2. Once ffmpeg is installed, install `pywhispercpp`

```shell
pip install pywhispercpp
```

If you want to use the examples, you will need to install extra dependencies

```shell
pip install pywhispercpp[examples]
```

Or install the latest dev version from GitHub

```shell
pip install git+https://github.com/abdeladim-s/pywhispercpp
```

# Quick start

```python
from pywhispercpp.model import Model

model = Model('base.en', n_threads=6)
segments = model.transcribe('file.mp3', speed_up=True)
for segment in segments:
    print(segment.text)
```

You can also assign a custom `new_segment_callback`

```python
from pywhispercpp.model import Model

model = Model('base.en', print_realtime=False, print_progress=False)
segments = model.transcribe('file.mp3', new_segment_callback=print)
```


* The `ggml` model will be downloaded automatically.
* You can pass any `whisper.cpp` [parameter](https://abdeladim-s.github.io/pywhispercpp/#pywhispercpp.constants.PARAMS_SCHEMA) as a keyword argument to the `Model` class or to the `transcribe` function.
* The `transcribe` function accepts any media file (audio/video), in any format.
* Check the [Model](https://abdeladim-s.github.io/pywhispercpp/#pywhispercpp.model.Model) class documentation for more details.


## Run app.py

## Errors
- If you encounter any issue with gr.Audio uncomment the commented line and comment the existing one.
- The existing works with gradio


## Functions
- transcribe: This function takes the audio input resample it using resamplr_to_16k function and saves it in  a temporary .wav file which will be deleted later. 
- resample_to_16k: This function resamples the speech rate of audio to 16k

## References:
- Main Github : [https://github.com/abdeladim-s/pywhispercpp/](https://github.com/abdeladim-s/pywhispercpp/)
- Documentation : [https://abdeladim-s.github.io/pywhispercpp/](https://abdeladim-s.github.io/pywhispercpp/)
