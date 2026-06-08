# Automated Skin Cancer Detection Using Deep Learning (CNN)

## Overview
Skin cancer is one of the most common and potentially life-threatening diseases worldwide. Early detection plays a critical role in improving survival rates, especially for melanoma. This project aims to develop an intelligent and efficient AI-based diagnostic support system capable of classifying skin lesions from dermoscopic images.
The system uses a lightweight yet powerful CNN model based on MobileNetV2, making it suitable for real-time and resource-constrained environments such as mobile healthcare applications. The project also includes a complete full-stack web application built using the MERN Stack for seamless image upload and prediction visualization.

## Features
- Deep Learning-based Skin Cancer Detection
- Multi-class Skin Lesion Classification
- MobileNetV2 Transfer Learning Architecture
- Focal Loss for Class Imbalance Handling
- Real-time Image Prediction System
- MERN Stack Full-Stack Web Application
- Responsive User Interface using React & Tailwind CSS
- MongoDB Database Integration
- REST API-based Backend Communication
- Optimized for Fast and Lightweight Deployment

## Technologies Used

### AI / Machine Learning
- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

### Web Development
- React.js
- Node.js
- Express.js
- MongoDB
- Tailwind CSS

## Dataset
The project uses the HAM10000 dataset, containing over 10,000 dermoscopic skin lesion images categorized into seven diagnostic classes:

- Melanocytic Nevi (nv)
- Melanoma (mel)
- Basal Cell Carcinoma (bcc)
- Benign Keratosis (bkl)
- Actinic Keratoses (akiec)
- Dermatofibroma (df)
- Vascular Lesions (vasc)

## Model Architecture
- MobileNetV2 as the backbone CNN architecture
- Transfer Learning using ImageNet pretrained weights
- Data Augmentation for improved generalization
- Focal Loss to handle severe dataset imbalance
- Softmax Classification Layer for multi-class prediction

## Model Performance
- Test Accuracy: 74.75%
- Weighted F1-Score: 0.72
- ROC-AUC (BCC): 0.95
- ROC-AUC (Melanoma): 0.82

## Future Improvements
- Integration of Explainable AI (XAI)
- Addition of patient metadata for multimodal learning
- Improved minority class detection
- Deployment on mobile devices
- Clinical-grade optimization and testing

## Author
Chetan Saini (B.Tech – Artificial Intelligence & Data Science)
Abhay Bansal (B.Tech – Artificial Intelligence & Data Science)

## Project Type
This project was developed as a collaborative academic project by the above team members.
