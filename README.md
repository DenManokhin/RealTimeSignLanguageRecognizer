# Real Time Sign Language Recognizer
## About
Educational project for real time recognition of American Sign Language.
Currently it has only 8 signs implemented (A, B, C, G, L, V, W, Y).
In future it is planned to add more gestures and provide hand localization system.

## Installation
Python3 is required.

1. Clone this repository
   ```bash
   git clone https://github.com/DenManokhin/RealTimeSignLanguageRecognizer.git
   ```
2. Install the required packages
   ```bash
   pip3 install -r requirements.txt
   ```
Peek inside the requirements file if you have everything already installed. Most of the dependencies are common libraries.

## How to use it
Run `main.py` to launch the application. Or `train.py` if you want tor train a model by yourself. 
Add `--help` as command line argument to explore available run parameters.
Training data is stored in `data` directory in csv files as flattened 28x28 images with labels.
