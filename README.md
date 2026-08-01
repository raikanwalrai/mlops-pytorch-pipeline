# 🚀 Cloud-Native MLOps Pipeline using PyTorch, Docker & Kubernetes

A complete cloud-native Machine Learning Operations (MLOps) pipeline for training, containerizing, deploying, serving, and validating an image classification model using **PyTorch**, **Docker**, **FastAPI**, and **Kubernetes**.

This repository was developed as part of **MLOps Assignment 3** for the Master of Technology (Artificial Intelligence) programme and demonstrates the end-to-end lifecycle of a production-style machine learning system—from model training to scalable deployment and validation on Kubernetes.

---

**Author:** Kanwal Rai

**Programme:** M.Tech (Artificial Intelligence)

**Repository Status:** ✅ Completed

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestrated-326CE5)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688)
![PyTest](https://img.shields.io/badge/PyTest-Tested-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Project Overview

Modern machine learning systems require considerably more than a well-performing model. Production deployments demand reproducibility, portability, scalability, monitoring, and reliable deployment workflows. These capabilities are collectively addressed through Machine Learning Operations (MLOps).

This project demonstrates the complete lifecycle of an image classification system using PyTorch, beginning with model training and progressing through Docker containerization, Kubernetes deployment, REST-based inference using FastAPI, persistent storage, health monitoring, and end-to-end validation.

The repository is designed not only to satisfy the assignment requirements but also to serve as a practical reference implementation for students and practitioners interested in cloud-native MLOps using open-source technologies.


## 📌 Project at a Glance

| Item | Description |
|------|-------------|
| Project | Cloud-Native MLOps Pipeline |
| Framework | PyTorch |
| Dataset | CIFAR-10 |
| Model | Convolutional Neural Network (CNN) |
| API Framework | FastAPI |
| Containerization | Docker |
| Orchestration | Kubernetes |
| Storage | Persistent Volume Claims |
| Configuration | ConfigMaps |
| Autoscaling | Horizontal Pod Autoscaler |
| Validation | End-to-End Kubernetes Deployment |
| Repository Status | ✅ Completed |


## 🏗️ System Architecture

The following architecture illustrates the complete cloud-native machine learning workflow implemented in this project.

<p align="center">
<img src="docs/architecture/architecture.png" width="900">
</p>

The workflow consists of four major stages:

1. Model Training
2. Docker Containerization
3. Kubernetes Deployment
4. REST-based Model Serving


## ✨ Features

- PyTorch CNN for CIFAR-10 image classification
- Modular project architecture
- Dockerized training and serving environments
- Kubernetes Job for model training
- FastAPI-based inference service
- Persistent storage using Kubernetes PVCs
- Centralized configuration using ConfigMaps
- Liveness, Readiness, and Startup Probes
- Rolling Update deployment strategy
- Horizontal Pod Autoscaler (HPA)
- Automated testing using PyTest
- End-to-end deployment verification


## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python 3.12 |
| Deep Learning Framework | PyTorch |
| Dataset | CIFAR-10 |
| API Framework | FastAPI |
| Containerization | Docker |
| Container Orchestration | Kubernetes |
| Testing Framework | PyTest |
| Configuration Management | ConfigMaps |
| Persistent Storage | Kubernetes Persistent Volume Claims (PVCs) |
| Autoscaling | Horizontal Pod Autoscaler (HPA) |
| Version Control | Git & GitHub |


## 📂 Repository Structure

```text
mlops-pytorch-pipeline/
│
├── configs/                  # Configuration files
├── docker/                   # Dockerfiles
├── docs/                     # Documentation and screenshots
│   ├── architecture/
│   ├── screenshots/
│   └── report/
├── k8s/                      # Kubernetes manifests
├── requirements/             # Python dependencies
├── src/                      # Source code
├── tests/                    # Unit tests
├── .gitignore
├── .dockerignore
├── README.md
└── image_1.jpg               # Sample image for inference
```

## 📋 Prerequisites

Before running the project, ensure the following software is installed:

- Python 3.12 or later
- Docker Desktop
- Kubernetes (Docker Desktop Kubernetes or Minikube)
- Git
- kubectl
- Docker CLI

The project was developed and validated using Windows 11 with WSL2 (Ubuntu), although it can also be executed on Linux and macOS with minimal modifications.

## 🚀 Quick Start

Clone the repository:

```bash
git clone https://github.com/raikanwalrai/mlops-pytorch-pipeline.git

cd mlops-pytorch-pipeline
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements/train.txt
pip install -r requirements/serve.txt
```


## 🏋️ Running the Training Pipeline

Train the CNN model locally:

```bash
python src/train.py
```

The training pipeline performs:

- Dataset loading
- Data preprocessing
- Model training
- Validation
- Checkpoint saving
- Metric logging

Trained model checkpoints are stored in the configured checkpoint directory.


## 🐳 Docker

Build the training image:

```bash
docker build \
-f docker/Dockerfile.train \
-t mlops-train:latest .
```

Build the serving image:

```bash
docker build \
-f docker/Dockerfile.serve \
-t mlops-serve:latest .
```

Both images were verified during the end-to-end validation phase.


## ☸️ Kubernetes Deployment

Deploy the application:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/checkpoints-pvc.yaml
kubectl apply -f k8s/data-pvc.yaml
kubectl apply -f k8s/training-job.yaml
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

Verify the deployment:

```bash
kubectl get all -n mlops-pipeline
```
## 🤖 Running Predictions

Forward the Kubernetes service:

```bash
kubectl port-forward service/pytorch-serving-service \
8081:80 \
-n mlops-pipeline
```

Verify the health endpoint:

```bash
curl http://localhost:8081/health
```

Example response:

```json
{
    "status":"healthy",
    "model":"loaded"
}
```

Run inference:

```bash
curl -X POST \
-F "file=@image_1.jpg" \
http://localhost:8081/predict
```

Example response:

```json
{
    "prediction":"automobile",
    "class_id":1,
    "confidence":0.5607
}
```

## 🤖 Running Predictions

Forward the Kubernetes service:

```bash
kubectl port-forward service/pytorch-serving-service \
8081:80 \
-n mlops-pipeline
```

Verify the health endpoint:

```bash
curl http://localhost:8081/health
```

Example response:

```json
{
    "status":"healthy",
    "model":"loaded"
}
```

Run inference:

```bash
curl -X POST \
-F "file=@image_1.jpg" \
http://localhost:8081/predict
```

Example response:

```json
{
    "prediction":"automobile",
    "class_id":1,
    "confidence":0.5607
}
```


## 📊 Project Validation

The complete cloud-native workflow was successfully validated.

| Validation Step | Status |
|-----------------|:------:|
| Docker Training Image | ✅ |
| Docker Serving Image | ✅ |
| Kubernetes Training Job | ✅ |
| Model Checkpoint Generation | ✅ |
| FastAPI Deployment | ✅ |
| Persistent Volumes | ✅ |
| ConfigMaps | ✅ |
| Health Endpoint | ✅ |
| Prediction Endpoint | ✅ |
| Horizontal Pod Autoscaler | ✅ |
| End-to-End Validation | ✅ |


## 📚 Documentation

The repository is accompanied by a comprehensive technical report covering:

- Project motivation
- System architecture
- Development methodology
- Docker implementation
- Kubernetes deployment
- Validation methodology
- Lessons learned
- Developer guide
- Troubleshooting guide
- Future work

Refer to the report for detailed implementation and design decisions.




## 🚀 Future Improvements

Potential future enhancements include:

- GPU-enabled Kubernetes training
- CI/CD using GitHub Actions
- MLflow experiment tracking
- Prometheus and Grafana monitoring
- Distributed training
- Model registry integration
- Canary deployments
- Blue-Green deployments
- Service mesh integration


## 🙏 Acknowledgements

This project builds upon several outstanding open-source technologies, including:

- PyTorch
- Docker
- Kubernetes
- FastAPI
- PyTest

The project documentation and engineering discussions also benefited from the use of OpenAI's ChatGPT as an AI-assisted software engineering companion for design reviews, documentation refinement, debugging discussions, and architectural reasoning. All implementation decisions, code validation, testing, and final verification were independently reviewed and executed by the author.


## 📄 License

This repository was developed as part of an academic M.Tech coursework submission and is intended for educational and research purposes.

