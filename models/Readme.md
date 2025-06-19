# 🤖 Pulmonitor AI Model (ResNet50)

- Framework: Keras (TensorFlow backend)
- Architecture: Transfer Learning with ResNet50 , VGG16, MobileNetV2, InceptionV3
- Input shape: 224x224 grayscale chest X-ray
- Output classes:
  - 0: Normal
  - 1: Pneumonia
- Accuracy: ~93.4%
- Format: `.h5` Keras SavedModel

Trained using Kaggle chest X-ray dataset (June 2025).
[Kaggle Chest X-ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)