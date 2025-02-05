# Sign-Language-Translation
American sign Language translation using mediapipe 

# Overview

This project implements real-time sign language detection using MediaPipe for hand, face, and pose detection and TensorFlow for action classification. It captures sign language gestures from a webcam and predicts actions based on a pre-trained model.

# Features

Real-time gesture recognition using a webcam

MediaPipe integration for pose, hand, and face landmark detection

Deep learning-based classification using an LSTM model

Expandable vocabulary (add more gestures with training)

User-friendly display of detected signs

Optimized for speed with minimal computational overhead

# Installation

# Requirements

Ensure you have the following dependencies installed:

pip install opencv-python numpy mediapipe tensorflow

Running the Program

Clone the repository:

git clone https://github.com/your-repo/sign-language-detection.git
cd sign-language-detection



# Usage

The program initializes the webcam and starts detecting gestures.

Detected actions will be displayed on the screen.

Press 'q' to exit.

Model Information

The model is trained on 30-frame sequences of hand, face, and pose landmarks.

Uses an LSTM-based neural network for sequence classification.

Currently supports: hello, thanks, iloveyou (expandable).





# License

MIT License