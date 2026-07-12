<div align="center">

# 🧠 EigenNoise
### Spectral Regularization for Neural Networks via Singular Value Control

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-blueviolet?style=flat-square)]()

*A regularization method that shapes the singular value spectrum of convolutional weights during training — improving generalization, conditioning, and robustness without the blunt instrument of L2 penalties.*

</div>

---

## 📖 Overview

Most regularization techniques (L2, dropout, weight decay) shrink weights uniformly, with no regard for the underlying **geometry** of what a layer is actually learning. **EigenNoise** takes a different approach: it directly monitors and constrains the **singular value spectrum** of convolutional kernels during training.

By penalizing singular values that drift too high (unstable, exploding directions) or too low (dead, uninformative directions), EigenNoise encourages convolutional filters to stay well-conditioned — learning representations that are neither redundant nor fragile.

This repository benchmarks EigenNoise against a **no-regularization baseline** and **standard L2 regularization** across two architectures:

- 🔹 Convolutional **Autoencoders** (CIFAR-10 reconstruction)
- 🔹 **ResNet-style CNNs** (CIFAR-10 classification)

---

## 💡 Motivation

> Why look at singular values at all?

Every convolutional kernel can be reshaped into a 2D matrix and decomposed via SVD:

```
W ∈ ℝ^(k·k·Cin × Cout)  →  W = U Σ Vᵀ
```

The singular values (Σ) describe how much a layer **stretches or compresses** information along different directions. A layer with:
- **Very large singular values** → unstable gradients, poor generalization
- **Very small singular values** → wasted capacity, collapsed feature directions

EigenNoise introduces a spectral penalty term that pulls singular values toward a healthy operating band, then measures the downstream effect on training dynamics, generalization, and the model's condition number.

---

## 🏗️ Architecture & Method

### 1️⃣ Baseline Models
Vanilla convolutional autoencoder and CNN trained with no regularization — used as the control group.

### 2️⃣ L2-Regularized Models
Standard weight decay (`kernel_regularizer=l2(1e-4)`) applied to convolutional layers for comparison.

### 3️⃣ EigenNoise Models
A custom `tf.keras.Model` subclass that:
1. Runs a standard forward/backward pass
2. Computes the SVD of each convolutional kernel
3. Applies a hinge-style penalty for singular values outside a target range `[σ_min, σ_max]`
4. Adds this spectral loss to the task loss (reconstruction / classification) before backpropagation

```python
def eigennoise_loss(model, sigma_min=0.5, sigma_max=5.0):
    loss = 0.0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            W = tf.reshape(layer.kernel, (-1, layer.kernel.shape[-1]))
            s = tf.linalg.svd(W, compute_uv=False)
            loss += tf.reduce_sum(tf.nn.relu(s - sigma_max) ** 2)
            loss += tf.reduce_sum(tf.nn.relu(sigma_min - s) ** 2)
    return loss
```

---

## 📊 Experiments

| Component | Details |
|---|---|
| **Dataset** | CIFAR-10 (60,000 32×32 color images, 10 classes) |
| **Task 1** | Autoencoder reconstruction (Baseline vs. L2 vs. EigenNoise) |
| **Task 2** | Image classification via ResNet-style CNN with residual blocks |
| **Metrics** | Training/validation loss, test accuracy, confusion matrix, per-class accuracy, average condition number of conv layers |
| **Analysis** | Singular value spectrum histograms across all three regimes, layer-by-layer |

### What's evaluated
- 📉 Training & validation loss curves across all three regimes
- 🔬 Singular value distribution histograms per convolutional layer
- 🧮 Average condition number (σ_max / σ_min) as a proxy for weight stability
- 🎯 Confusion matrices & classification reports
- 🖼️ Misclassified sample inspection

> Full metrics, plots, and printed results are generated live in the notebook — see [`Eigen_noise_maths.ipynb`](./Eigen_noise_maths.ipynb) for complete output.

---

## 📁 Project Structure

```
eigennoise/
│
├── Eigen_noise_maths.ipynb    # Main notebook: data, models, training, analysis
├── README.md                  # You are here
└── requirements.txt           # Dependencies
```

---

## ⚙️ Installation & Usage

### Prerequisites
```bash
python >= 3.10
```

### Setup
```bash
git clone https://github.com/yakkalakarthikeya/eigennoise.git
cd eigennoise
pip install -r requirements.txt
```

### Run
Open the notebook in Jupyter or Google Colab:
```bash
jupyter notebook Eigen_noise_maths.ipynb
```

> 💡 Recommended: run on Colab with a GPU runtime — the ResNet-CIFAR10 experiment (80 epochs) benefits significantly from acceleration.

### Requirements
```
tensorflow>=2.12
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🧭 Key Takeaways

- Spectral regularization offers a more **geometry-aware** alternative to L2 weight decay
- Constraining singular values directly influences the **conditioning** of learned filters
- The approach generalizes across both **reconstruction** (autoencoder) and **classification** (CNN) tasks

---

## 🚧 Future Work

- [ ] Extend spectral penalty to fully-connected and attention layers
- [ ] Benchmark on larger datasets (CIFAR-100, TinyImageNet)
- [ ] Compare against spectral normalization (Miyato et al.) as an additional baseline
- [ ] Ablation study on `σ_min` / `σ_max` sensitivity

---

## 👤 Author

**Karthikeya Yakkala**
B.Tech, Artificial Intelligence & Data Science — Amrita Vishwa Vidyapeetham
IEEE Published Author · AI/ML Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-yakkalakarthikeya-181717?style=flat-square&logo=github)](https://github.com/yakkalakarthikeya)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.

---

<div align="center">

*If you found this project interesting, consider giving it a ⭐ — it helps others discover it too!*

</div>
