# MLOps Example
In this repo you can find some examples of useful packages for MLOps. Then we will put it all togheter to create a simple, but complete, MLOps pipeline.

### Table of Contents

1. [Installation](#installation)
2. [Demo](#demo)

## Installation <a name="installation"></a>

You will need Miniconda3 and Python to follow these instructions. You will also need a linux environment or WSL for Windows users.

1. Create a virtual environment
    ```console
    conda create --name mlops-example python=3.9
    ```

2. Activate environment
    ```console
    conda activate mlops-example
    ```

3. Install requirements
    ```console
    pip install -r requirements.txt
    ```

## Demo<a name="demo"></a>

1. Optuna

    Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning.

    Run command bellow and follow JupyterLab code to see how optuna works.

    ```console
    mlflow run packages/optuna --env-manager=local
    ```

    Open optuna.ipynb

2. MLFlow

    MLflow is an open source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry.

    In packages/mlflow there is a `training.py` file. It's a super simple machine learning training code. The command above call it's code and tracks the ML training job in MLFlow.

    ```console
    mlflow run packages/mlflow --env-manager=local
    ```

    To see track info:

    ```console
    mlflow ui
    ```

    In case it fails, run to clear localhost port:
    ```console
    fuser -k 5000/tcp
    ```

3. Hydra

    A framework for elegantly configuring complex applications.

    In packages/hydra there is a `config.yaml` file. It's a configuration file for hydra parameters. Note that `training.py` is the same code we have in step 2 with some changes to work with hydra. To run the training code with hydra parameters use:

    ```console
    mlflow run packages/hydra --env-manager=local
    ```

4. All together

    Now we will put all together and in the root directory we have the final codes and parameters, note that `config.yaml` has changed to use optuna inside hydra, so now we can train a pipeline and tune hyperparameters from command line and track it on MLFlow.

    ```console
    mlflow run . --env-manager=local
    ```