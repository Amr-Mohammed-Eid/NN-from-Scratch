<div align="center">

# 🧠 Neural Network from Scratch

**A pure-NumPy implementation of a two-layer neural network trained on the MNIST handwritten-digit dataset.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Overview

This project demonstrates how to build a fully functional neural network **entirely from scratch**, using only **NumPy** — no PyTorch, no TensorFlow, no Keras. Every piece of the learning pipeline — forward propagation, back-propagation, gradient descent — is implemented manually to solidify a deep intuition for how deep learning really works under the hood.

The model is trained on the classic **MNIST digit-recognition dataset** (Kaggle format) and achieves strong validation accuracy in just a few hundred iterations.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Architecture** | `784 → 10 (ReLU) → 10 (Softmax)` |
| **Weight Init** |Initialization (`√2/n`) |
| **Activations** | ReLU (hidden), Softmax (output) |
| **Loss** | Cross-Entropy |
| **Optimizer** | Vanilla Gradient Descent |
| **Framework** | Pure NumPy — zero DL libraries |
| **Visualization** | Loss & accuracy curves + sample prediction grid |

---

## 🗂️ Project Structure

```
NN from Scratch/
│
├── NN.ipynb              # Main notebook — full pipeline end-to-end
│
└── Data/
    ├── train.csv         # MNIST training set  (Kaggle format)
    ├── test.csv          # MNIST test set
    └── sample_submission.csv
```

---

## 🏗️ Architecture

```
Input Layer        Hidden Layer       Output Layer
  (784)        →     (10, ReLU)    →   (10, Softmax)
  28×28 px           He Init            Digit 0–9
```

The network is intentionally kept shallow to demonstrate that even a single hidden layer, trained from scratch, can learn meaningful digit representations.

---

## 🔬 Implementation Walkthrough

### 1 — Data Loading & Preprocessing
- Load `train.csv` with Pandas, shuffle with a fixed seed (`42`) for reproducibility  
- Split into **1 000-sample dev set** and the remainder as the **training set**  
- Normalise pixel values to `[0, 1]` by dividing by 255

### 2 — Parameter Initialisation
```python
W1 = np.random.randn(10, 784) * np.sqrt(2.0 / 784)   # He init
W2 = np.random.randn(10,  10) * np.sqrt(2.0 /  10)
b1 = np.zeros((10, 1))
b2 = np.zeros((10, 1))
```

### 3 — Forward Propagation
```
Z1 = W1 · X + b1    →    A1 = ReLU(Z1)
Z2 = W2 · A1 + b2   →    A2 = Softmax(Z2)
```

### 4 — Backward Propagation (analytical gradients)
```
dZ2 = A2 − one_hot(Y)
dW2 = (1/m) dZ2 · A1ᵀ        db2 = (1/m) Σ dZ2
dZ1 = (W2ᵀ · dZ2) ⊙ ReLU'(Z1)
dW1 = (1/m) dZ1 · Xᵀ         db1 = (1/m) Σ dZ1
```

### 5 — Training Loop
- **Iterations**: 1 000  
- **Learning rate**: 0.1  
- Logs loss, train accuracy, and validation accuracy every 50 iterations

---

## 📊 Results

After 1 000 iterations of gradient descent the network converges to:

| Metric | Value |
|---|---|
| Final Validation Accuracy | **~92 %** |
| Loss Function | Cross-Entropy |
| Training Samples | ~41 000 |
| Validation Samples | 1 000 |

> Exact accuracy will vary slightly due to random shuffling, but consistently exceeds 85 % on the dev set.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib jupyter
```

### Dataset

1. Download the MNIST dataset from [Kaggle – Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data)
2. Place `train.csv`, `test.csv`, and `sample_submission.csv` inside the `Data/` folder

### Run

```bash
jupyter notebook NN.ipynb
```

Execute cells top-to-bottom. Training takes ~30–60 seconds on a modern CPU.

---

## 📈 Visualisations

The notebook automatically plots:

- **Cross-Entropy Loss** vs. iteration  
- **Train & Validation Accuracy** vs. iteration  
- **Sample prediction grid** — 10 random dev-set images with predicted vs. true labels (green = correct ✅, red = wrong ❌)

---

## 🧩 Key Concepts Demonstrated

- ✅ Forward & backward propagation implemented analytically  
- ✅ He weight initialisation to prevent vanishing/exploding gradients  
- ✅ Numerically stable Softmax (`Z − max(Z)`)  
- ✅ One-hot encoding for multi-class cross-entropy  
- ✅ Train / validation split for honest evaluation  
- ✅ Gradient descent parameter update rule  

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open a [Pull Request](../../pulls) or file an [Issue](../../issues).

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

Made with ❤️ and pure NumPy &nbsp;|&nbsp; No deep-learning frameworks harmed in the making of this project.

</div>
