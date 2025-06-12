# 🧠 Breast Cancer Prediction with Neural Networks (PyTorch)

> A binary classification model that predicts whether a tumor is malignant or benign using a simple feedforward neural network. Achieves **~98% test accuracy** on the UCI Breast Cancer dataset.

---

## 🧩 Problem & Impact

Breast cancer is one of the most common cancers in women worldwide. Early and accurate diagnosis plays a critical role in treatment outcomes. This project aims to assist diagnostic decisions by building a machine learning model that predicts malignancy based on tumor features, potentially reducing reliance on costly or invasive procedures.

---

## 📊 Dataset

- **Source**: [UCI Breast Cancer Wisconsin Diagnostic dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- **Features**: 30 numeric features derived from digital images of fine needle aspirates of breast masses.
- **Samples**: 569 total (212 malignant, 357 benign)
- **Preprocessing**:
  - Standardized all features with `StandardScaler`
  - Split: 80% training / 20% test

---

## 🧠 Model Architecture

| Layer        | Details                            |
|--------------|-------------------------------------|
| Input        | 30 features                         |
| Hidden       | 1 layer with 64 neurons + ReLU      |
| Output       | 1 neuron + Sigmoid (binary output)  |
| Loss Function| Binary Cross-Entropy (BCELoss)      |
| Optimizer    | Adam (`lr=0.001`)                   |
| Epochs       | 100                                 |

---

## 📈 Results

| Dataset | Accuracy |
|---------|----------|
| Train   | ~98.4%   |
| Test    | ~98.2%   |

The model performs with high precision and recall on unseen data, making it suitable as a decision-support tool in clinical settings.

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/HadisZare12/Breast-Cancer-Prediction.git
cd Breast-Cancer-Prediction

# 2. Install dependencies
pip install torch scikit-learn matplotlib

# 3. Run the notebook
jupyter notebook Breast_Cancer_Prediction.ipynb
