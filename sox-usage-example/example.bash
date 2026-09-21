#! /bin/bash

# install sox and ffmpeg
sudo apt install sox
sudo apt install ffmpeg

# convert m4a to wav and create spectrogram
ffmpeg -i input.m4a -f wav - | sox - -n spectrogram -o output.png

# OR create spectrogram directly from wav file
sox input.wav -n spectrogram -o output.png