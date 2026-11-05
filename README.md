# Mammogram Abnormality Detection using CNN and VGG16 Transfer Learning

![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)

![Workflow](assets/DEEPmammo_Breast_Cancer_Detection_Workflow.jpg)

A deep learning project that classifies mammogram images from the CBIS-DDSM dataset as **benign** or **malignant**. A custom CNN trained from scratch is compared against a VGG16 transfer-learning model.

> **Academic project. Not a medical device and not for clinical use.**

---

## Project Overview

Breast cancer is one of the leading causes of cancer-related deaths among women worldwide, and early detection through mammography improves outcomes. Manual interpretation of mammograms is time-consuming and subject to inter-observer variability.

This project builds a complete pipeline, from loading the CBIS-DDSM metadata to training and evaluating two models, and asks a simple question: does ImageNet transfer learning help on mammography compared with a small CNN trained from scratch?

- **Model 1:** custom CNN trained from scratch on 50 × 50 images
- **Model 2:** VGG16 (ImageNet weights, frozen backbone) with a new classifier head on 150 × 150 images

---

## Dataset

**CBIS-DDSM** (Curated Breast Imaging Subset of the Digital Database for Screening Mammography).

The notebook reads the JPEG version of the dataset together with its metadata CSVs (`dicom_info.csv` and the calcification/mass case description files for the train and test sets). The paths in the notebook (`/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/...`) assume it is run on Kaggle with that dataset attached.

- Original source: https://www.cancerimagingarchive.net/collection/cbis-ddsm/
- The dataset is not included in this repository because of its size and licensing terms.

---

## Methodology

1. **Data loading:** load `dicom_info.csv` and the four case-description CSVs; build image paths for full mammogram images.
2. **Label building:** merge pathology labels (benign / malignant) with image paths using the patient ID.
3. **Preprocessing:** load each image, convert to 3-channel RGB, resize, and scale pixels from 0–255 to 0–1.
4. **Class balancing:** the larger class is randomly undersampled to the size of the smaller one, giving equal benign and malignant counts (295 of each in the test set).
5. **Split:** 80% training / 20% testing, stratified, `random_state=42`; labels are one-hot encoded.
6. **Augmentation:** `ImageDataGenerator` with rotation (20°), width/height shift (10%), shear (10%), zoom (10%) and horizontal flip.
7. **Training:** Adam optimizer, categorical cross-entropy, up to 50 epochs, with `EarlyStopping` (patience 5, restore best weights) and `ReduceLROnPlateau` (factor 0.2, patience 2), both monitoring validation loss.

| Model | Input size | Batch size |
|---|---|---|
| Custom CNN | 50 × 50 × 3 | 32 |
| VGG16 | 150 × 150 × 3 | 16 |

---

## Model Architecture

![Architecture Comparison](assets/Breast_Cancer_Detection_Architecture_Comparison.jpg)

### Model 1: Custom CNN

Four convolution blocks followed by a small classifier head:

- 4 × [Conv2D (32 → 64 → 128 → 128 filters, 3×3, ReLU) → BatchNormalization → MaxPooling]
- Flatten → Dense (128, ReLU) → Dropout (0.3) → Dense (2, softmax)
- About 0.39 million parameters

**Training performance**

![CNN Training](assets/cnn_training.png)

**Confusion matrix**

![CNN Confusion Matrix](assets/cnn_confusion_matrix.png)

**Sample prediction**

![CNN Prediction](assets/cnn_prediction.png)

### Model 2: VGG16 Transfer Learning

- VGG16 convolutional base with ImageNet weights, all layers frozen
- New head: Flatten → Dense (256, ReLU) → Dropout (0.5) → Dense (2, softmax)
- About 14.7 million frozen parameters and 2.1 million trainable parameters (head only). The backbone is not fine-tuned.

**Training performance**

![VGG16 Training](assets/vgg16_training.png)

**Confusion matrix**

![VGG16 Confusion Matrix](assets/vgg16_confusion_matrix.png)

**Sample prediction**

![VGG16 Prediction](assets/vgg16_prediction.png)

---

## Results

Evaluated on the held-out test split of 590 images (295 per class).

| Metric | Custom CNN | VGG16 |
|---|---|---|
| Test accuracy | 60% | 62% |
| Precision (Benign) | 0.70 | 0.64 |
| Recall (Benign) | 0.34 | 0.57 |
| F1-score (Benign) | 0.46 | 0.60 |
| Precision (Malignant) | 0.56 | 0.61 |
| Recall (Malignant) | 0.86 | 0.67 |
| F1-score (Malignant) | 0.68 | 0.64 |
| Best epoch (0-indexed) | 19 | 14 |

**Interpretation**

- The two models reach similar overall accuracy. A 2-point difference on 590 test images is within normal run-to-run variation, so this project does not show that transfer learning is clearly better here.
- The custom CNN is biased toward predicting malignant: high malignant recall (0.86) but low benign recall (0.34), meaning many benign cases are flagged as malignant (195 of 295).
- VGG16 is more balanced across the two classes (recall 0.57 benign, 0.67 malignant).
- A plausible reason for the modest VGG16 gain is the mismatch between ImageNet natural images and mammograms, together with the low input resolution. This was not tested here.
- Neither model is accurate enough for clinical use.

---

## Model Comparison

| Feature | Custom CNN | VGG16 Transfer Learning |
|---|---|---|
| Architecture | 4-block Sequential CNN with BatchNorm | Frozen VGG16 backbone + dense head |
| Input size | 50 × 50 | 150 × 150 |
| Total parameters | about 0.39 M | about 16.8 M (about 2.1 M trainable) |
| Pretrained weights | None | ImageNet |
| Test accuracy | 60% | 62% |

---

## Limitations

- **Test set used during training:** the same 20% split is used as the validation set for early stopping and learning-rate scheduling, and the best-epoch weights are restored from it. Reported test numbers may therefore be slightly optimistic. A separate validation split would give cleaner estimates.
- **Patient-level labels:** labels are assigned per patient ID (duplicates dropped), while a patient can have more than one lesion. Some images may carry an imperfect label.
- **Split by image, not by patient:** images from the same patient can appear in both training and test sets.
- **Low resolution:** images are downsized to 50 × 50 and 150 × 150, which discards fine detail that matters in mammography.
- **Single run:** results come from one training run per model, with no repeated runs or confidence intervals.

---

## Installation

```bash
git clone https://github.com/prawinm-engineer/mammogram-abnormality-detection.git
cd mammogram-abnormality-detection
pip install -r requirements.txt
```

## Usage

1. Obtain the CBIS-DDSM dataset (the notebook is written for the Kaggle JPEG version).
2. Update `dicom_csv_path`, `image_dir` and the CSV paths in the notebook if you are not running on Kaggle.
3. Run `deepmammo-cnn-vgg16.ipynb` from top to bottom to train and evaluate both models.

Note: the notebook is saved without cell outputs. The figures in `assets/` come from the original training runs.

---

## Project Structure

```
mammogram-abnormality-detection/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── deepmammo-cnn-vgg16.ipynb
└── assets/
    ├── DEEPmammo_Breast_Cancer_Detection_Workflow.jpg
    ├── Breast_Cancer_Detection_Architecture_Comparison.jpg
    ├── cnn_training.png
    ├── cnn_confusion_matrix.png
    ├── cnn_prediction.png
    ├── vgg16_training.png
    ├── vgg16_confusion_matrix.png
    └── vgg16_prediction.png
```

---

## Future Improvements

- Hold out a separate validation set and split by patient
- Use higher input resolution and try EfficientNet, ResNet or DenseNet
- Fine-tune the top VGG16 blocks instead of freezing the whole backbone
- Add Grad-CAM for model interpretability
- Hyperparameter search and repeated runs with confidence intervals

---

## References

1. CBIS-DDSM dataset: https://www.cancerimagingarchive.net/collection/cbis-ddsm/
2. TensorFlow documentation: https://www.tensorflow.org/
3. Keras documentation: https://keras.io/
4. Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* https://arxiv.org/abs/1409.1556

---

## Author

Prawin M, Biomedical Engineering
