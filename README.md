# Trash Classifier Model

This repository contains the training, evaluation, and conversion pipeline for a machine learning model designed to classify trash into six categories using images. The model is built with TensorFlow/Keras for deployment in a Flutter application via TensorFlow Lite.

## 📦 Model Overview

- **Architecture**: Transfer learning using MobileNetV2 as the feature extractor.
- **Classes**: `compost`, `garbage`, `glass`, `recycling-paper`, `recycling-plastic`, `hazardous-waste`
- **Input Shape**: `(224, 224, 3)`
- **Output**: Softmax probabilities for 6 categories
- **Deployment Format**: `.tflite` model for Flutter integration
