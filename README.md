# Facial Recognition using PCA, LDA, and KNN

This project implements a facial recognition system using dimensionality reduction techniques — Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) — followed by classification using the K-Nearest Neighbors (KNN) algorithm. The model distinguishes between face and non-face images using datasets such as the ORL face dataset and the CIFAR-10 dataset.

---

## 📁 Dataset
- Faces: ORL face dataset (images resized and reshaped to 10304-dimensional vectors)
- Non-Faces: [CIFAR-10 dataset](https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders)

### Data Preparation
- Data loaded into NumPy arrays.
- Labels: 1 for face, 0 for non-face.
- Splits used:
    - 50% train / 50% test
    - 70% train / 30% test
- Face images fixed at 200.
- Non-face images tested with varying sizes: [50, 200, 500, 1000, 3000]

---

## 🔍 Techniques
1. PCA (Principal Component Analysis)
   - Used for unsupervised dimensionality reduction.
   - Extracted Eigenfaces.
   - Explored variance retention values: [0.8, 0.85, 0.9, 0.95]
   - Resulting dimensions: [36, 51, 76, 115]
   - Visualized top 6 eigenvectors (Eigenfaces).
2. KNN (K-Nearest Neighbors)
   - Final classifier used after PCA/LDA projection.
   - Tested with different numbers of neighbors (k).
   - Evaluated performance on fixed test set.
3. LDA (Linear Discriminant Analysis)
   - Used for supervised dimensionality reduction.
   - Calculated per-class means and scatter matrices.
   - Used top eigenvectors for projection (1 eigenvector for binary classification).
   - Applied to both 50/50 and 70/30 splits.

---

## 📊 Experiments & Results

### Face vs. Non-Face Classification
- Fixed test data: 200 images
- Training with different numbers of non-face samples
- Accuracy tested for each alpha using KNN
- PCA reached ~84% accuracy
- Kernel PCA had inconsistent accuracy

### Bonus Experiments
- PCA Variants:
  - Built-in PCA
  - Randomized PCA
  - Kernel PCA
  - Incremental PCA
- LDA Variants (only KDA worked):
  - FDA – Fisher's Discriminant Analysis
  - QDA – Quadratic Discriminant Analysis
  - KDA – Kernel Discriminant Analysis
  - DLA – Diagonal Discriminant Analysis

---

## 🧪 Confusion Matrices
Confusion matrices were created to analyze performance for different non-face sample sizes using LDA and KNN.

---

## 📌 Key Notes
- PCA is unsupervised and reduces dimensionality by maximizing variance.
- LDA is supervised and reduces dimensionality by maximizing class separability.
- Combining PCA or LDA with KNN improves classification performance.


