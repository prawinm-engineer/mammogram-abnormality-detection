# Mammogram-Based Breast Abnormality Detection

## Overview
This project explores a deep learning–based approach for classifying mammogram
images as **Benign** or **Malignant**. The objective is to study how convolutional
neural networks can be applied to medical image data for early abnormality detection.

This is an academic project developed as part of biomedical engineering coursework.

## Dataset
CBIS-DDSM (Curated Breast Imaging Subset of DDSM)

> The dataset is not included in this repository due to size and licensing restrictions.

## Methodology
• Data preprocessing and normalization of mammogram images  
• Dataset balancing to reduce class bias  
• Model 1: Custom CNN trained from scratch (50×50 images)  
• Model 2: VGG16 using transfer learning (150×150 images)  
• Performance comparison using accuracy and confusion matrix  

## Evaluation Metrics
• Accuracy  
• Precision, Recall, F1-score  
• Confusion Matrix  

## Key Observation
The transfer learning model (VGG16), although powerful, did not significantly
outperform the simple CNN due to **domain mismatch** between natural images
(ImageNet) and medical X-ray images.

## Tools & Technologies
• Python  
• TensorFlow / Keras  
• OpenCV  
• NumPy, Pandas, Matplotlib  
• Scikit-learn  

## Disclaimer
This project is for **academic and research purposes only**  
and is **not intended for clinical diagnosis or medical use**.

## Author
Prawin M – Biomedical Engineering
