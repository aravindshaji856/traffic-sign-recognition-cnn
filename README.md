# Traffic Sign Recognition using CNN

An end-to-end image classification pipeline that recognizes traffic signs — **stop**, **yield**, and **speed limit** — using a Convolutional Neural Network built with TensorFlow/Keras.

## Overview
This project covers the full ML workflow: preprocessing raw images, training a CNN from scratch, evaluating performance, and serializing the trained model for reuse.

## Dataset
The dataset consists of 150 images per class (450 total), organized as:
dataset/
├── speed_limit/
├── stop/
└── yield/

**Note:** This is a synthetic dataset generated to enable building and testing the full pipeline in the absence of a labeled real-world dataset. As a result, the model has not been validated on real-world photos, which would introduce more variation in lighting, angle, and background than the training data contains.

## Pipeline
1. **Preprocessing** — resize to 32×32, convert to grayscale, normalize pixel values
2. **Data split** — 70% train / 15% validation / 15% test (stratified)
3. **Model** — a small CNN (2 Conv2D + MaxPooling blocks, Dense layer, Dropout, softmax output)
4. **Training** — with early stopping and light data augmentation (rotation, brightness)
5. **Evaluation** — classification report and confusion matrix on the held-out test set
6. **Serialization** — trained model saved and reloaded via Pickle
7. **Inference demo** — sample predictions with per-image confidence and latency

## Tech Stack
Python, TensorFlow/Keras, OpenCV, scikit-learn, NumPy, Matplotlib

## How to Run
```bash
pip install tensorflow opencv-python scikit-learn numpy matplotlib
jupyter notebook traffic_sign_recognition.ipynb
```
Run all cells in order. Requires the `dataset/` folder to be present in the project root.

## Results
See the classification report and confusion matrix output in the notebook for exact metrics on the test set.

## Limitations
- Trained and evaluated only on synthetic images, not real-world photos
- Only 3 classes covered, not a full traffic sign taxonomy
- Small dataset size (150 images/class)
