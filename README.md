# MLOps PyTorch Pipeline

A production-style machine learning pipeline for training and serving a PyTorch image classification model using Docker and Kubernetes.

## Project Overview

This project demonstrates the complete deployment lifecycle of a PyTorch image classification workload:

* Local development with Git and GitHub
* PyTorch model training
* Docker-based training and inference
* Kubernetes-based training using Jobs
* Kubernetes-based model serving
* Configuration using YAML and Kubernetes ConfigMaps
* Health checks and scalable model serving

The model will be trained on the CIFAR-10 image classification dataset.

## Project Structure

```text
mlops-pytorch-pipeline/
├── README.md
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   └── serve.py
├── configs/
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   ├── configmap.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
└── tests/
    └── test_model.py
```

## Architecture

```text
                    ┌─────────────────────┐
                    │     Developer       │
                    │  Git / GitHub PRs   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   GitHub Actions    │
                    │        CI           │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐        ┌──────────────────┐
       │ Docker Training  │        │ Docker Serving   │
       │     Image        │        │      Image       │
       └────────┬─────────┘        └────────┬─────────┘
                │                           │
                ▼                           │
       ┌──────────────────┐                 │
       │ Kubernetes Job   │                 │
       │ PyTorch Training │                 │
       └────────┬─────────┘                 │
                │                           │
                ▼                           │
       ┌──────────────────┐                 │
       │ Model Checkpoint │─────────────────┘
       │ Persistent Store │
       └──────────────────┘
                                            │
                                            ▼
                                  ┌──────────────────┐
                                  │ Kubernetes       │
                                  │ Deployment       │
                                  │ 2 Serving Pods   │
                                  └────────┬─────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │ Kubernetes       │
                                  │ Service :80      │
                                  │ → Container :8080│
                                  └──────────────────┘
```

## Technologies

* Python
* PyTorch
* Torchvision
* Docker
* Kubernetes
* Git
* GitHub Actions
* Flask or FastAPI for model serving

## Development Workflow

The project follows a Git-based development workflow.

```text
main
  │
  └── develop
        │
        ├── feature/project-setup
        ├── feature/pytorch-model
        ├── feature/docker-training
        └── feature/k8s-deployment
```

All feature work is performed on feature branches and merged through Pull Requests.

## Local Development

### Prerequisites

The project requires:

* Python 3.10 or newer
* Docker Desktop or a Docker-enabled environment
* kubectl
* A Kubernetes cluster such as Minikube, kind, or a cloud-managed cluster
* A GitHub account

### Clone the repository

```bash
git clone https://github.com/da25m520/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
```

### Python environment

A Python virtual environment can be created for local development:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Dependencies will be installed from the appropriate files in the `requirements/` directory.

## Training

The training workload will read its configuration from:

```text
configs/training_config.yaml
```

The trained model checkpoint will be stored in the configured checkpoint directory.

## Docker

The project contains separate Dockerfiles for:

* Training
* Model serving

Training image:

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

Serving image:

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

## Kubernetes

The Kubernetes resources are stored in the `k8s/` directory.

The workflow includes:

1. Creating the `ml-training` namespace
2. Creating the training configuration
3. Running model training as a Kubernetes Job
4. Deploying the model-serving application
5. Exposing the application through a Kubernetes Service
6. Configuring horizontal scaling

## Model API

The serving application will expose:

```text
GET  /health
POST /predict
```

`/health` is used by Kubernetes health probes.

`/predict` accepts an image and returns class probabilities.

## Status

This project is being developed incrementally as part of the MLOps & Infrastructure for Machine Learning assignment.
