# 🏭 Industrial Steel Defect Detection

### MLOps & Deep Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Classification-green)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)

An end-to-end **industrial computer vision and MLOps project** for detecting and classifying steel surface defects using deep learning.

The project covers the complete machine learning lifecycle: **data preparation, model development, experimentation, evaluation, API serving and containerized deployment**.

---

## 📌 Project Snapshot

| Component           | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| **Problem**         | Multi-class industrial steel surface defect classification |
| **Dataset**         | NEU Metal Surface Defect Database                          |
| **Task**            | Image Classification                                       |
| **Approaches**      | Custom CNN + ResNet50 Transfer Learning                    |
| **Evaluation**      | Accuracy, Precision, Recall, F1-Score, Confusion Matrix    |
| **Experimentation** | Baseline vs. improved model comparison                     |
| **Serving**         | FastAPI REST API                                           |
| **Deployment**      | Docker                                                     |
| **Focus**           | Computer Vision + MLOps                                    |

---

## 🎯 The Problem: Why This Project Exists?

Manual inspection of industrial steel surfaces is time-consuming and can be affected by human subjectivity.

Automated defect detection can help manufacturers:

* Detect defects consistently and automatically
* Reduce dependence on manual inspection
* Improve production quality and reliability
* Support faster quality-control decisions
* Create a foundation for automated industrial inspection systems

The goal of this project is therefore not only to train an image classifier, but to demonstrate how a deep learning model can be transformed into a **deployable ML application**.

---

# 🗂️ Dataset: NEU Metal Surface Defects

The project uses the **NEU Metal Surface Defect Database**, a benchmark dataset for industrial steel surface defect classification.

The dataset contains six defect categories:

1. **Crazing (Cr)**
2. **Inclusion (In)**
3. **Patches (Pa)**
4. **Pitted Surface (Ps)**
5. **Rolled-in Scale (Rs)**
6. **Scratches (Sc)**

The six-class structure makes the dataset suitable for evaluating both classical CNN architectures and transfer-learning approaches.

---

## 🔧 Data Preprocessing

Before training, the images are prepared through a consistent preprocessing pipeline.

Key steps include:

* Image resizing and normalization
* Train/validation/test dataset separation
* Data augmentation for improved generalization
* Consistent preprocessing between training and inference
* Label encoding for the six defect classes

The preprocessing pipeline is designed to prevent data leakage and ensure that the model receives images in a consistent format during both training and deployment.

---

# 🧠 Model Architecture & Strategy

Two different approaches were investigated to understand the trade-off between a custom architecture and transfer learning.

## 1. Custom CNN — Lightweight Baseline

A custom convolutional neural network was developed as a baseline model.

Key objectives:

* Build a model specifically for the defect-classification problem
* Establish a baseline performance
* Keep the architecture relatively lightweight
* Understand how a CNN learns industrial surface patterns

The custom CNN provides a useful reference point for evaluating whether a pretrained architecture can provide a meaningful performance improvement.

---

## 2. ResNet50 Transfer Learning — High-Capacity Model

A pretrained **ResNet50** architecture was used as a second approach.

Transfer learning was selected because pretrained convolutional networks already contain useful low-level and mid-level visual representations such as:

* Edges
* Textures
* Shapes
* Local patterns

These representations can be adapted to industrial defect images through fine-tuning.

### Why ResNet50?

ResNet50 provides a strong balance between:

* Feature extraction capability
* Model depth
* Training stability
* Classification performance
* Practical deployment potential

The final classification layer was adapted to the six defect classes in the NEU dataset.

---

# 📊 A/B Testing & Performance Evaluation

Rather than evaluating a single model in isolation, the project compares different approaches experimentally.

The main objective is to determine whether the more advanced architecture provides a meaningful improvement over the baseline.

### Evaluation Metrics

The models are evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

Using multiple metrics is important because accuracy alone may hide class-specific weaknesses.

For example, a model can achieve high overall accuracy while performing poorly on one particular defect class.

---

## 📈 Training & Validation Results

The training curves are used to monitor:

* Training performance
* Validation performance
* Convergence behaviour
* Potential overfitting
* Differences between experimental configurations

![Training Results](assets/training_results.png)

---

## ⚖️ Model Comparison

The experiments compare the baseline and improved architectures in terms of predictive performance and model characteristics.

![Model Comparison](assets/model_comparison.png)

The comparison helps determine which architecture provides the best overall trade-off between predictive performance and deployment considerations.

---

# 🔍 Error Analysis — Confusion Matrix

The confusion matrix provides a class-level view of model behaviour.

It helps identify:

* Which defect classes are classified correctly
* Which classes are frequently confused
* Whether visually similar defects create classification difficulties
* Where future data collection or augmentation could provide the most value

![Confusion Matrix](assets/confusion_matrix.png)

### Why Confusion Matrix?

Industrial defect classes can have visually similar texture and structural characteristics.

Therefore, a confusion matrix provides much more useful information than a single accuracy value when analysing model weaknesses.

---

# 🚀 MLOps Pipeline: FastAPI + Docker

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

This structure separates the **model development layer** from the **serving and deployment layer**.

---

## 1. FastAPI Inference API

The trained model is exposed through a REST API using FastAPI.

The API is responsible for:

1. Receiving an input image
2. Applying the same preprocessing used during training
3. Running model inference
4. Returning the predicted defect class
5. Returning the prediction confidence

Example response:

```json
{
  "prediction": "Pitted Surface",
  "confidence": 0.94
}
```

This makes the model usable by external applications rather than requiring direct interaction with the training code.

---

## 2. Containerization with Docker

The inference application is containerized using Docker.

Benefits include:

* Reproducible deployment environment
* Dependency isolation
* Easier local deployment
* Consistent runtime behaviour
* Easier integration into larger ML systems

The objective is to make the trained model portable from the development environment to a production-like inference environment.

---

# 🐳 Quick Start with Docker

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

## 2. Build the Docker Image

```bash
docker build -t steel-defect-detector .
```

## 3. Run the Container

```bash
docker run -p 8000:8000 steel-defect-detector
```

The API should then be available at:

```text
http://localhost:8000
```

FastAPI's interactive documentation can be accessed through:

```text
http://localhost:8000/docs
```

---

# 📡 API Usage

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

The exact endpoint and response structure depend on the implementation in the API module.

---

# 📁 Project Structure

```text
industrial-steel-defect-detection/
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   └── trained_model.pth
│
├── notebooks/
│   └── experiments.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── api/
│   └── main.py
│
├── assets/
│   ├── training_results.png
│   ├── model_comparison.png
│   └── confusion_matrix.png
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🧪 Reproducibility

The project is structured so that the same preprocessing, model and inference logic can be reproduced across environments.

Important reproducibility considerations include:

* Fixed preprocessing pipeline
* Defined dataset splits
* Versioned dependencies
* Saved model weights
* Containerized inference environment

For future iterations, experiment tracking and automated model versioning can further improve reproducibility.

---

# 💡 Key Engineering Decisions

### Why CNN?

Steel surface defects are strongly characterized by local texture, edges and spatial patterns. CNNs are therefore naturally suited to extracting hierarchical visual features from defect images.

### Why Transfer Learning?

Training a deep architecture completely from scratch can require more data and computational resources. Transfer learning allows the project to reuse general visual representations learned from a large-scale dataset.

### Why Multiple Metrics?

Accuracy alone does not reveal class-specific weaknesses. Precision, recall, F1-score and the confusion matrix provide a more complete evaluation.

### Why FastAPI?

The trained model needs an interface that external applications can consume. FastAPI provides a lightweight REST-based serving layer for model inference.

### Why Docker?

Docker makes the inference environment reproducible and isolates application dependencies, making the model easier to deploy and integrate.

---

# 🔮 Future Improvements

Potential next steps include:

* [ ] Experiment tracking with MLflow
* [ ] Automated CI/CD pipeline
* [ ] Model versioning
* [ ] Automated data validation
* [ ] Model monitoring
* [ ] Inference latency benchmarking
* [ ] Explainable AI with Grad-CAM
* [ ] More extensive hyperparameter optimization
* [ ] Automated retraining pipeline
* [ ] Cloud deployment

---

# 🎯 Project Goal

This project demonstrates an end-to-end approach to industrial computer vision:

```text
Data
  ↓
Preprocessing
  ↓
Deep Learning
  ↓
Experimentation
  ↓
Evaluation
  ↓
Error Analysis
  ↓
FastAPI
  ↓
Docker
  ↓
Deployable ML Application
```

The main objective is not simply to achieve a high classification score, but to demonstrate how a computer vision model can be developed, evaluated and transformed into a **reproducible and deployable machine learning system**.

---

## 👨‍💻 Technologies

**Programming & ML**

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn

**Computer Vision**

* OpenCV
* Image preprocessing
* CNNs
* Transfer Learning
* ResNet50

**MLOps & Deployment**

* FastAPI
* Docker
* REST API

**Evaluation**

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Training/Validation Curves
