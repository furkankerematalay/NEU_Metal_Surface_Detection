# 🏭 Industrial Steel Defect Detection (MLOps & Deep Learning Pipeline)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-cyan.svg)](https://www.docker.com/)

An end-to-end computer vision pipeline for detecting and classifying surface defects in industrial steel using deep learning.

### 📌 Project Snapshot
*   **Task:** Multi-class defect classification (6 classes)
*   **Models:** Custom CNN (Baseline) vs. ResNet50 (Transfer Learning)
*   **Dataset:** NEU Surface Defect Database (1,800 images)
*   **Deployment:** FastAPI + Docker (Containerized REST API)
*   **Evaluation:** Accuracy, Precision, Recall, F1-Score, A/B Testing, Confusion Matrix

---

## 🚀 5-Second Summary
This project replaces slow, inconsistent human visual inspection on the factory floor with a high-speed, automated Deep Learning pipeline. It evaluates a lightweight Custom CNN against a heavy-duty ResNet50 model, ultimately deploying the optimal solution as a Dockerized FastAPI endpoint for real-time inference.

---

## 📊 Results & Model Comparison

A key focus of this project was engineering a solution that balances classification accuracy with inference speed.

| Model | Architecture Type | Accuracy | F1-Score | Parameters | Inference Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A** | Custom CNN (Baseline) | ~88% | ~87% | ~1.2M | **High** (Edge-optimized) |
| **Model B** | ResNet50 (Transfer Learning) | **~96%** | **~96%** | ~23.6M | Moderate |

**Engineering Decision:**
> *Model B (ResNet50) improved overall F1-Score by ~9% compared to the baseline. While it requires more parameters, the increase in inference latency is negligible for our specific factory use-case. Therefore, ResNet50 was selected as the production candidate.*

*(Interactive A/B test dashboards and Confusion Matrices are available in the `reports/` directory).*

![ResNet50 Confusion Matrix](reports/resnet_confusion_matrix.png)

---

## 📖 The Problem & Dataset

*   **The Problem:** Surface defects (crazing, inclusions, scratches, etc.) lead to structural weaknesses. Manual inspection bottlenecks the production line and is subject to human fatigue.
*   **The Dataset:** We utilized the benchmark NEU Surface Defect Database, structuring the raw flat files into a standardized (Train/Val/Test) pipeline using Keras `image_dataset_from_directory`. Pixel values were normalized specifically for ResNet50 architecture requirements.

---

## 🧠 Methodology: Why These Architectures?

Steel surface defects are characterized by local textures, edges, and complex spatial patterns. CNNs are structurally appropriate because their convolutional filters can learn these hierarchical visual representations directly from defect images.

*   **Why Custom CNN (Baseline)?:** To establish a minimum viability threshold and evaluate performance on low-compute edge devices.
*   **Why ResNet50?:** To leverage deep feature extraction without vanishing gradients (via residual connections). The base layers were frozen (transfer learning from ImageNet), and a custom classification head (GlobalAveragePooling -> Dense -> Dropout) was attached to prevent overfitting on our smaller dataset.

---

## ⚙️ MLOps Architecture

This project goes beyond a Jupyter Notebook. It implements a production-ready deployment pipeline.

```text
Dataset --> Preprocessing --> Model Training --> Evaluation (A/B Test) --> FastAPI --> Docker --> Inference API
