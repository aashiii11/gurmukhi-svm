# SVM-based Handwritten Gurmukhi Character Recognition

## Overview
This project implements a handwritten Gurmukhi character recognition system using a Support Vector Machine (SVM) classifier. It covers dataset exploration, image preprocessing, feature extraction, model training, evaluation, and misclassification analysis.

## Dataset
- Source: [Gurmukhi Dataset – Mendeley Data](https://data.mendeley.com/datasets/h65gdk4ptv/1)
- 41 character classes, ~12,128 total images
- Balanced dataset (~291–299 images per class)
- Image size: 256x256 RGB

## Pipeline
1. **Preprocessing**: Grayscale conversion → Otsu thresholding (binarization) → Resize to 64x64 → Normalize to [0,1]
2. **Feature Extraction**: Histogram of Oriented Gradients (HOG), producing a 1764-dimensional feature vector per image
3. **Model**: SVM with RBF kernel (C=10, gamma='scale'), trained using scikit-learn
4. **Train/Test Split**: 80/20, stratified by class

## Results
- **Test Accuracy: 87.47%**
- Macro F1-score: 0.87
- Strongest classes: 41, 9, 8, 1 (F1 ≥ 0.92)
- Weakest classes: 18, 3, 13 (F1 ≤ 0.76)

## Misclassification Analysis
Errors are not random — they cluster around visually similar character pairs, e.g.:
- Class 21 ↔ 18 (7 misclassifications)
- Class 3 ↔ 15 (6 misclassifications)

Both pairs share nearly identical stroke structure (a top horizontal stroke followed by a loop-and-tail formation), suggesting the SVM is learning genuine shape-based features. Improving performance on these pairs likely requires finer-grained features or additional targeted training data.

## Tech Stack
- Python, OpenCV, scikit-image, scikit-learn, NumPy, Matplotlib, Seaborn
- Developed and trained on Google Colab

## Reproducing
1. Download the dataset from the Mendeley link above
2. Run `gurmukhi_svm.ipynb` end to end (mount Google Drive, update dataset path)
3. Trained model and features can optionally be cached as `.npy`/`.pkl` files for faster re-runs
