# CIFAR-10 Dataset Classification
This project explores the CIFAR-10 dataset, implementing both an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN) for image classification. The goal is to compare the performance of these models on the CIFAR-10 dataset and achieve a high level of accuracy.

## Overview
Dataset: CIFAR-10 is a well-known dataset consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images.

## Objective:
The objective of this project is to build and compare an ANN and a CNN to classify these images into their respective classes.

## Implementation
### Artificial Neural Network (ANN)
- The ANN consists of three fully connected (Dense) layers with ReLU activation.
- The output layer uses the sigmoid activation function for binary classification.
### Convolutional Neural Network (CNN)
- The CNN consists of multiple convolutional layers followed by max-pooling layers and then fully connected layers.
- ReLU activation is used throughout the CNN, and softmax is used in the output layer for multi-class classification.

## Training and Evaluation
- Both the ANN and CNN are trained using the CIFAR-10 training dataset.
- The models are evaluated using the CIFAR-10 testing dataset.
- Performance is evaluated based on accuracy, precision, recall, and F1-score.

## Results
- The ANN achieved an accuracy of 47% on the CIFAR-10 testing dataset.
- The CNN achieved an accuracy of 70% on the CIFAR-10 testing dataset.
- The CNN outperformed the ANN, demonstrating the effectiveness of convolutional neural networks for image classification tasks.

## Conclusion
- This project serves as a project for the image classification task.
- It demonstrates the difference in performance between ANN and CNN models on the CIFAR-10 dataset.
- Further optimization and tuning of the models could potentially improve performance.

## Future Work
- Experiment with different architectures and hyperparameters to further improve performance.
- Explore data augmentation techniques to enhance the dataset and improve generalization.
- Investigate other state-of-the-art models and techniques for image classification tasks.
