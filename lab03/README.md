# Experiment and Model Management with MLflow

## What is MLFLow?

In their own words:

> MLflow is an open-source platform, purpose-built to assist machine learning practitioners and teams in handling the complexities of the machine learning process. MLflow focuses on the full lifecycle for machine learning projects, ensuring that each phase is manageable, traceable, and reproducible.

MLFLow is an end-to-end platform with several components and be used for much more than Experiment management:

- [Tracking][1]: An API and UI for logging parameters, code versions, metrics, and artifacts.
- [Evaluate][2]: `mlflow.evaluate` provides an API to evaluate the performance of a model on one or more datasets. It supports classification, regression, and a range of language modeling tasks. Evaluations are logged to the tracking API.
- [Model Registry][3]: An API and UI for storing and managing different versions of models.
- [Projects][4]: MLflow Projects are a standardized format for packaging ML code, workflows, and artifacts.
- [MLflow deployments for LLMs][5]: A server that provides a set of standardized APIs for streamlining access to both closed- and open-source LLMs.
- [Prompt Engineering UI][6]: A dedicated UI for experimenting, refining, and evaluating prompts.
- [Recipes][7]: Blueprints for structuring (real-world) ML projects and deployments.

In this lab, we will primarily be using the Tracking. Other components follow in future Labs.

## The MLflow UI

Let's start a local instance of MLFLow (run this in a terminal):

```shell
mlflow server --host 127.0.0.1 --port 8080
```

This will start an MLFlow tracking server, its UI and all the other necessary components. Use your browser to navigate to `localhost:8080`. You should be presented with a page that looks like the one in the screenshot below:

![MLFlow UI](imgs/image.png)

As you can see, there are two tabs: one for experiments and one for models. In the subsequent sections, we will introduce the various components, so keep the MLflow instance running.

##  What you will learn

| Topic                 | Notebook |
|-----------------------|----------|
| Experiment tracking with MLflow  | [Tracking](mlflow_tracking.ipynb) |
| Hyperparameter tuning with  Ray Tune | [Tuning](ray_tune.ipynb) ([Solution](ray_tune_solution.ipynb))|
| Hyperparameter tuning a teeny tiny diffusion model| [Diffusing](diffusion.ipynb)|

[1]: https://mlflow.org/docs/latest/tracking.html#tracking
[2]: https://mlflow.org/docs/latest/models.html#model-evaluation
[3]: https://mlflow.org/docs/latest/model-registry.html#registry
[4]: https://mlflow.org/docs/latest/projects.html#projects
[5]: https://mlflow.org/docs/latest/llms/deployments/index.html#deployments
[6]: https://mlflow.org/docs/latest/llms/prompt-engineering/index.html#prompt-engineering
[7]: https://mlflow.org/docs/latest/recipes.html#recipes
