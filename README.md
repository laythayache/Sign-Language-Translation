# Sign Language Detection & Translation App

## Overview

This project presents a real-time sign language detection and translation application utilizing Python, OpenCV, MediaPipe, and TensorFlow. The application employs a custom-trained model to recognize common sign language gestures—such as "hello," "thanks," and "I love you"—and translates them into text, facilitating seamless communication for the Deaf community.

## Key Features

- **Real-Time Processing:** Captures and processes video input from a webcam in real-time.
- **Gesture Recognition:** Utilizes MediaPipe to extract keypoints (pose, face, and hand landmarks) and a TensorFlow model to classify sign language gestures.
- **Custom LSTM Compatibility:** Implements a custom LSTM layer (`MyLSTM`) to address compatibility issues with the `time_major` parameter.
- **Visual Feedback:** Displays live video with annotated landmarks and overlay text indicating the recognized gesture.
- **Versatile Applications:** Suitable for video calls, emergency communication, educational environments, customer service, and more.

## Prerequisites

- Python 3.7 or higher
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [MediaPipe](https://mediapipe.dev/)
- [TensorFlow](https://www.tensorflow.org/) 2.x

Install the required packages using `pip`:

```bash
pip install opencv-python mediapipe tensorflow numpy

```
# how to run:
just run the American expressions.py file and make sure you have access to your webcam 


### Note: some problem that might occur could be the "LSTM time_major" error, this happens because different TensorFlow and Keras implementations of LSTM handle the input tensor shape differently, so i just told tensorflow to ignore the argument, everything should be clear in the code 

# Future Improvements 
i want to add in the future a Nothing output so that when it detects none of the signs it will not just be stable to one, 
i also have another repository that contains the alphabet and how i trained them, that i want to mix with these.
