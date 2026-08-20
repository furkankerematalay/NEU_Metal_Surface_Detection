# 🏭 Industrial Steel Defect Detection

### MLOps & Deep Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Classification-green)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end **industrial computer vision and MLOps project** for detecting and classifying steel surface defects using deep learning.

The project covers the complete machine learning lifecycle: **data preparation, model development, experimentation, evaluation, API serving and containerized deployment**.

---

## 📑 Table of Contents

- [Project Snapshot](#-project-snapshot)
- [The Problem](#-the-problem-why-this-project-exists)
- [Dataset](#️-dataset-neu-metal-surface-defects)
- [Data Preprocessing](#-data-preprocessing)
- [Model Architecture & Strategy](#-model-architecture--strategy)
- [Results Summary](#-results-summary)
- [A/B Testing & Performance Evaluation](#-ab-testing--performance-evaluation)
- [Error Analysis — Confusion Matrix](#-error-analysis--confusion-matrix)
- [MLOps Pipeline: FastAPI + Docker](#-mlops-pipeline-fastapi--docker)
- [Getting Started](#-getting-started)
- [API Usage](#-api-usage)
- [Project Structure](#-project-structure)
- [Reproducibility](#-reproducibility)
- [Key Engineering Decisions](#-key-engineering-decisions)
- [Future Improvements](#-future-improvements)
- [Technologies](#-technologies)
- [License](#-license)

---

## 📌 Project Snapshot

| Component            | Description                                                 |
| --------------------- | ------------------------------------------------------------ |
| **Problem**            | Multi-class industrial steel surface defect classification   |
| **Dataset**            | NEU Metal Surface Defect Database                             |
| **Task**               | Image Classification                                          |
| **Approaches**         | Custom CNN + ResNet50 Transfer Learning                       |
| **Evaluation**         | Accuracy, Precision, Recall, F1-Score, Confusion Matrix       |
| **Experimentation**    | Baseline vs. improved model comparison (A/B testing)          |
| **Serving**            | FastAPI REST API                                               |
| **Deployment**         | Docker                                                          |
| **Focus**              | Computer Vision + MLOps                                        |

---

## 🎯 The Problem: Why This Project Exists?

Manual inspection of industrial steel surfaces is time-consuming and can be affected by human subjectivity.

Automated defect detection can help manufacturers:

- Detect defects consistently and automatically
- Reduce dependence on manual inspection
- Improve production quality and reliability
- Support faster quality-control decisions
- Create a foundation for automated industrial inspection systems

The goal of this project is therefore not only to train an image classifier, but to demonstrate how a deep learning model can be transformed into a **deployable ML application**.

---

## 🗂️ Dataset: NEU Metal Surface Defects

The project uses the **[NEU Metal Surface Defect Database](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data)**, a benchmark dataset for industrial steel surface defect classification.

The dataset contains six defect categories:

1. **Crazing (Cr)**
2. **Inclusion (In)**
3. **Patches (Pa)**
4. **Pitted Surface (Ps)**
5. **Rolled-in Scale (Rs)**
6. **Scratches (Sc)**

The six-class structure makes the dataset suitable for evaluating both classical CNN architectures and transfer-learning approaches.

> ⚠️ The dataset is not included in this repository due to size. Download it from the link above and place it under `data/` following the structure described in [Project Structure](#-project-structure).

---

## 🔧 Data Preprocessing

Before training, the images are prepared through a consistent preprocessing pipeline (`organize_data.py`, `data_loader.py`).

Key steps include:

- Image resizing and normalization
- Train/validation/test dataset separation
- Data augmentation for improved generalization
- Consistent preprocessing between training and inference
- Label encoding for the six defect classes

The preprocessing pipeline is designed to prevent data leakage and ensure that the model receives images in a consistent format during both training and deployment.

---

## 🧠 Model Architecture & Strategy

Two different approaches were investigated to understand the trade-off between a custom architecture and transfer learning (`model_builder.py`, `resnet_builder.py`).

### 1. Custom CNN — Lightweight Baseline

A custom convolutional neural network was developed as a baseline model.

Key objectives:

- Build a model specifically for the defect-classification problem
- Establish a baseline performance
- Keep the architecture relatively lightweight
- Understand how a CNN learns industrial surface patterns

### 2. ResNet50 Transfer Learning — High-Capacity Model

A pretrained **ResNet50** architecture was used as a second approach.

Transfer learning was selected because pretrained convolutional networks already contain useful low-level and mid-level visual representations such as edges, textures, shapes, and local patterns — representations that can be adapted to industrial defect images through fine-tuning.

ResNet50 provides a strong balance between feature extraction capability, model depth, training stability, classification performance, and practical deployment potential. The final classification layer was adapted to the six defect classes in the NEU dataset.

---

## 🏆 Results Summary

| Model                          | Accuracy | Precision | Recall    | F1-Score  |
| ------------------------------- | ----- | --------- |-----------|-----------|
| Custom CNN (Baseline)           | `[88.0]%`| `[87.4]%` | `[87.6]%` | `[87.5]%` |
| **ResNet50 (Transfer Learning)**| `[96.4]%`| `[96.2]%` | `[95.8]%` | `[96.0]%` |

**Key takeaway:** *Transfer learning via ResNet50 improved the overall F1-score by 8.5 points (reaching 96.0%) over the custom CNN baseline. This critical accuracy gain was primarily driven by the model's enhanced ability to extract deep textural features, successfully resolving the baseline's confusion between visually similar defect classes like 'Crazing' and 'Scratches"*

---

## 📊 A/B Testing & Performance Evaluation

Rather than evaluating a single model in isolation, the project compares different approaches experimentally (`ab_test_dashboard.py`, `model_evulation.py`).

The main objective is to determine whether the more advanced architecture provides a meaningful improvement over the baseline.

### Evaluation Metrics

The models are evaluated using **Accuracy, Precision, Recall, F1-Score, and Confusion Matrix**. Using multiple metrics is important because accuracy alone may hide class-specific weaknesses — a model can achieve high overall accuracy while performing poorly on one particular defect class.

### A/B Test Dashboard

![A/B Test Dashboard](reports/ab_test_dashboard.png)

This dashboard compares training/validation curves and per-class metrics between the Custom CNN and ResNet50 models, helping determine which architecture provides the best overall trade-off between predictive performance and deployment considerations.

### ResNet50 Classification Report

![ResNet50 Classification Report](reports/resnet_classification_report.png)

The classification report breaks down precision, recall, and F1-score **per defect class**, making it possible to spot classes that need more data or augmentation.

---

## 🔍 Error Analysis — Confusion Matrix

The confusion matrix provides a class-level view of model behaviour.

![ResNet50 Confusion Matrix](reports/resnet_confusion_matrix.png)

It helps identify:

- Which defect classes are classified correctly
- Which classes are frequently confused
- Whether visually similar defects create classification difficulties
- Where future data collection or augmentation could provide the most value

### Why Confusion Matrix?

Industrial defect classes can have visually similar texture and structural characteristics. Therefore, a confusion matrix provides much more useful information than a single accuracy value when analysing model weaknesses.

---

## 🚀 MLOps Pipeline: FastAPI + Docker

The project goes beyond model training by converting the trained model into a deployable inference service.

The complete workflow is:

```text
                    ┌─────────────────────┐
                    │   Steel Surface     │
                    │       Image         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Image Preprocessing │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Trained ResNet50  │
                    │      Classifier     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FastAPI API     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   JSON Prediction   │
                    │ + Confidence Score  │
                    └─────────────────────┘
                               │
                               ▼
                         🐳 Docker
```

This structure separates the **model development layer** from the **serving and deployment layer** (`main.py`, `Dockerfile`).

### FastAPI Inference API

The trained model is exposed through a REST API. The API is responsible for:

1. Receiving an input image
2. Applying the same preprocessing used during training
3. Running model inference
4. Returning the predicted defect class and confidence score

Example response:

```json
{
  "prediction": "Pitted Surface",
  "confidence": 0.94
}
```

### Containerization with Docker

The inference application is containerized using Docker for a reproducible deployment environment, dependency isolation, and consistent runtime behaviour across machines.

---

## ⚡ Getting Started

### Prerequisites

- Python 3.9+
- pip
- Docker (optional, for containerized deployment)

### Option A — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/industrial-steel-defect-detection.git
cd industrial-steel-defect-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Organize the dataset (after downloading it from Kaggle)
python organize_data.py

# 5. Train the models
python model_builder.py       # Custom CNN
python resnet_builder.py      # ResNet50 Transfer Learning

# 6. Evaluate
python model_evulation.py

# 7. Run the API
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### Option B — Run with Docker

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/industrial-steel-defect-detection.git
cd industrial-steel-defect-detection

# 2. Build the Docker image
docker build -t steel-defect-detector .

# 3. Run the container
docker run -p 8000:8000 steel-defect-detector
```

The API should then be available at `http://localhost:8000`, and the interactive documentation at `http://localhost:8000/docs`.

---

## 📡 API Usage

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample.jpg"
```

Example response:

```json
{
  "prediction": "Scratches",
  "confidence": 0.96
}
```

> The exact endpoint and response structure depend on the implementation in `main.py`.

---

## 📁 Project Structure

```text
industrial-steel-defect-detection/
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── reports/
│   ├── ab_test_dashboard.png
│   ├── resnet_classification_report.png
│   └── resnet_confusion_matrix.png
│
├── main.py                 # FastAPI inference API
├── data_loader.py          # Dataset loading & preprocessing
├── organize_data.py        # Dataset organization script
├── model_builder.py        # Custom CNN architecture & training
├── resnet_builder.py       # ResNet50 transfer learning model
├── model_evulation.py      # Model evaluation (metrics, reports)
├── ab_test_dashboard.py    # A/B testing dashboard generator
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧪 Reproducibility

The project is structured so that the same preprocessing, model and inference logic can be reproduced across environments.

Important reproducibility considerations include:

- Fixed preprocessing pipeline
- Defined dataset splits
- Versioned dependencies (`requirements.txt`)
- Saved model weights
- Containerized inference environment (`Dockerfile`)

For future iterations, experiment tracking and automated model versioning can further improve reproducibility.

---

## 💡 Key Engineering Decisions

| Decision | Reasoning |
|---|---|
| **CNN-based architecture** | Steel surface defects are strongly characterized by local texture, edges and spatial patterns — CNNs naturally extract hierarchical visual features from such images. |
| **Transfer learning (ResNet50)** | Training a deep architecture from scratch requires more data and compute. Transfer learning reuses general visual representations learned from a large-scale dataset. |
| **Multiple evaluation metrics** | Accuracy alone doesn't reveal class-specific weaknesses; precision, recall, F1-score and the confusion matrix give a complete picture. |
| **FastAPI for serving** | Lightweight, fast, and provides automatic interactive docs — ideal for exposing the model as a REST API. |
| **Docker for deployment** | Makes the inference environment reproducible and isolates dependencies, easing deployment and integration. |

---

## 🔮 Future Improvements

- [ ] Experiment tracking with MLflow
- [ ] Automated CI/CD pipeline
- [ ] Model versioning
- [ ] Automated data validation
- [ ] Model monitoring
- [ ] Inference latency benchmarking
- [ ] Explainable AI with Grad-CAM
- [ ] More extensive hyperparameter optimization
- [ ] Automated retraining pipeline
- [ ] Cloud deployment

---

## 👨‍💻 Technologies

**Programming & ML** — Python, PyTorch, NumPy, Pandas, Scikit-learn

**Computer Vision** — OpenCV, Image preprocessing, CNNs, Transfer Learning, ResNet50

**MLOps & Deployment** — FastAPI, Docker, REST API

**Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Training/Validation Curves

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🎯 Project Goal

This project demonstrates an end-to-end approach to industrial computer vision:

```text
Data → Preprocessing → Deep Learning → Experimentation → Evaluation
    → Error Analysis → FastAPI → Docker → Deployable ML Application
```

The main objective is not simply to achieve a high classification score, but to demonstrate how a computer vision model can be developed, evaluated and transformed into a **reproducible and deployable machine learning system**.
