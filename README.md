# MLOps PyTorch Pipeline

A production-style MLOps pipeline for training and serving a PyTorch image classification model using GitHub Actions, Docker, and Kubernetes.

The project uses the CIFAR-10 dataset and demonstrates an incremental development workflow using Git feature branches and Pull Requests.

## Project Overview

This project demonstrates the lifecycle of a machine learning workload:

- Local development with Python and PyTorch
- CIFAR-10 dataset loading and preprocessing
- PyTorch CNN model training
- Configuration-driven training
- Model checkpoint generation
- FastAPI-based model serving
- Docker-based training and serving
- Kubernetes-based training using Jobs
- Kubernetes-based model serving
- Kubernetes ConfigMaps
- Health checks
- Horizontal Pod Autoscaling
- Automated testing with pytest
- Continuous Integration using GitHub Actions

## Project Structure

```text
mlops-pytorch-pipeline/
|
+-- .github/
|   +-- workflows/
|       +-- ci.yml
|
+-- checkpoints/
|   +-- classifier_v1.pt
|
+-- configs/
|   +-- training_config.yaml
|
+-- docker/
|   +-- Dockerfile.train
|   +-- Dockerfile.serve
|
+-- k8s/
|   +-- namespace.yaml
|   +-- training-job.yaml
|   +-- serving-deployment.yaml
|   +-- serving-service.yaml
|   +-- configmap.yaml
|   +-- hpa.yaml
|
+-- requirements/
|   +-- train.txt
|   +-- serve.txt
|
+-- src/
|   +-- dataset.py
|   +-- model.py
|   +-- train.py
|   +-- serve.py
|
+-- tests/
|   +-- test_model.py
|
+-- .gitignore
+-- README.md
```

## Architecture

```text
                    Developer
                       |
                       v
                 Git / GitHub
                       |
                       v
                GitHub Actions CI
                       |
             +---------+---------+
             |                   |
             v                   v
      Docker Training      Docker Serving
          Image                Image
             |                   |
             v                   |
      Kubernetes Job             |
             |                   |
             v                   |
       Model Checkpoint          |
             |                   |
             +---------+---------+
                       |
                       v
              Kubernetes Deployment
                       |
                +------+------+
                |             |
                v             v
             Pod 1          Pod 2
                |             |
                +------+------+
                       |
                       v
              Kubernetes Service
                       |
                       v
                 FastAPI API
                    :8080
```

## Technologies

- Python
- PyTorch
- Torchvision
- FastAPI
- Uvicorn
- Docker
- Kubernetes
- Git
- GitHub
- GitHub Actions
- pytest
- YAML

## Git Development Workflow

The project follows a feature-branch development workflow.

```text
main
 |
 +-- develop
      |
      +-- feature/project-setup
      |
      +-- feature/pytorch-model
      |
      +-- feature/docker
      |
      +-- feature/repository-documentation
```

All feature development is performed on separate branches and merged into `develop` through Pull Requests.

The `main` branch represents the stable project branch.

## Configuration

Training parameters are maintained in:

```text
configs/training_config.yaml
```

Example configuration:

```yaml
model:
  architecture: simple_cnn
  num_classes: 10

training:
  epochs: 10
  batch_size: 64
  learning_rate: 0.001
  early_stopping_patience: 3
```

Keeping training parameters in a YAML configuration file makes the training process easier to reproduce and modify without changing the training code.

## Local Development

### Prerequisites

The project requires:

- Python 3.10 or newer
- Git
- Docker Desktop
- kubectl
- A Kubernetes environment such as Minikube, kind, or another Kubernetes cluster

### Clone the repository

```bash
git clone https://github.com/da25m520/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
```

### Create a Python environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required dependencies using the appropriate requirements file.

## Testing

The project uses pytest for automated model testing.

Run:

```bash
python -m pytest -v
```

The model forward-pass test verifies that the implemented model can successfully process an input tensor and produce the expected output.

## Training

The training implementation is located at:

```text
src/train.py
```

The training process uses:

```text
src/dataset.py
src/model.py
configs/training_config.yaml
```

The model uses the CIFAR-10 dataset for image classification.

The trained model is saved as a PyTorch checkpoint under:

```text
checkpoints/
```

The checkpoint filename is configured through:

```text
configs/training_config.yaml
```

## Docker

The project contains separate Dockerfiles for training and model serving.

### Training image

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

The training image contains the training dependencies and application source code.

The training workload can use the CIFAR-10 dataset and produces a model checkpoint.

### Serving image

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

The serving image contains the FastAPI application and the model checkpoint required for inference.

Docker-based execution is part of the project's deployment workflow.

## Kubernetes

Kubernetes resources are maintained in the `k8s/` directory.

The configured deployment workflow consists of:

1. Creating the `ml-training` namespace
2. Applying the training configuration
3. Running model training using a Kubernetes Job
4. Producing the model checkpoint
5. Deploying the model-serving application
6. Exposing the serving application through a Kubernetes Service
7. Configuring health probes
8. Configuring Horizontal Pod Autoscaling

The Kubernetes manifests include resources for:

- Namespace
- Training Job
- Serving Deployment
- Serving Service
- ConfigMap
- Horizontal Pod Autoscaler

## Model API

The model-serving application is implemented using FastAPI.

The serving application provides a health endpoint and a prediction endpoint.

The API is configured to run on:

```text
8080
```

The serving application includes:

```text
GET  /
POST /predict
```

The root endpoint provides a basic health response.

The prediction endpoint accepts feature input and returns the model prediction.

## CI Pipeline

GitHub Actions is used to automatically validate the project.

The CI workflow is located at:

```text
.github/workflows/ci.yml
```

The CI workflow provides automated project validation, including running the test suite.

## Development and Deployment Flow

The intended workflow is:

```text
Developer
    |
    v
Feature Branch
    |
    v
Pull Request
    |
    v
develop
    |
    v
GitHub Actions CI
    |
    +----------------------+
    |                      |
    v                      v
Training                Serving
    |                      |
    v                      v
Docker / Kubernetes    Docker / Kubernetes
    |                      |
    v                      v
Model Checkpoint       FastAPI Service
```

This separation allows model development, packaging, and deployment infrastructure to be developed incrementally.

## Current Status

The project is being developed incrementally as part of the MLOps & Infrastructure for Machine Learning assignment.

The repository currently contains:

- Project repository structure
- Git feature-branch workflow
- GitHub Actions CI workflow
- CIFAR-10 dataset handling
- PyTorch CNN model
- Configuration-driven training pipeline
- Model checkpoint generation
- FastAPI model-serving application
- Docker training configuration
- Docker serving configuration
- Kubernetes deployment manifests
- Automated model testing
- Project documentation

Docker and Kubernetes components are included in the repository and are being validated incrementally as part of the deployment stages.
