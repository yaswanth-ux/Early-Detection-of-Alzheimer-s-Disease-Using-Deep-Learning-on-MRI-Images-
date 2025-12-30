# Early-Detection-of-Alzheimer-s-Disease-Using-Deep-Learning-on-MRI-Images-
The primary objective of this project is to explore and compare deep learning and transfer learning approaches for accurate and efficient early detection of Alzheimer’s Disease, highlighting the trade-offs between model complexity, training time, and classification performance.

📌 Overview

Alzheimer’s Disease (AD) is a progressive neurological disorder that affects memory and cognitive abilities. Early diagnosis plays a critical role in slowing disease progression and improving patient care. This project applies deep learning and transfer learning techniques to brain MRI images to detect Alzheimer’s Disease at early stages.

🎯 Objectives

To classify brain MRI images into different cognitive impairment stages

To design a custom CNN model optimized for medical imaging

To apply transfer learning using pre-trained deep learning models

To compare models based on accuracy, loss, training time, and complexity

🧠 Dataset

Modality: Brain MRI images

Format: Grayscale images converted to RGB for transfer learning

Classes:

No Impairment

Very Mild Impairment

Mild Impairment

Moderate Impairment

Preprocessing:

Image resizing and normalization

Data augmentation for better generalization

🏗️ Model Architectures
1. Custom Convolutional Neural Network (CNN)

Designed specifically for MRI data

Fewer parameters and faster convergence

Achieves high validation accuracy

2. AlexNet (Transfer Learning)

Pre-trained on ImageNet

Convolutional layers frozen

Used as a feature extractor

3. ResNet50 (Transfer Learning)

Deep residual network with skip connections

ImageNet pre-trained weights

Feature extraction with frozen backbone

⚙️ Methodology

Data loading and preprocessing

Model construction (Custom CNN, AlexNet, ResNet50)

Transfer learning using pre-trained weights

Training with early stopping

Evaluation on validation data

Performance comparison and visualization

📊 Results
Model	Validation Accuracy	Validation Loss	Training Time	Parameters
Custom CNN	98.25%	0.0737	1671.84 s	1.13 M
AlexNet (TL)	82.00%	0.4198	394 s	46.7 M
ResNet50 (TL)	72.82%	0.6407	3112 s	24.6 M
📈 Visualizations

Validation accuracy comparison

Validation loss comparison

Training time comparison

Model size (parameters) comparison

Performance trend curves

🛠 Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy, Pandas

Matplotlib


Evaluation Metrics

Validation Accuracy

Validation Loss

Training Time

Number of Trainable Parameters

🔍 Key Observations

Custom CNN performs best due to domain-specific design

Transfer learning models benefit from pre-trained features but suffer from domain mismatch

Model complexity does not always guarantee higher accuracy in medical imaging

⚠️ Limitations

MRI modality only (no PET or multimodal data)

Transfer learning models trained on natural images

🔮 Future Work

Fine-tuning deeper layers of transfer learning models

Using 3D CNNs for volumetric MRI data

Multimodal learning (MRI + PET)

Deployment using a web-based interface
